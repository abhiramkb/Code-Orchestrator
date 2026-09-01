# How the parallel inner loop works

Implementation notes for commit `422b00d` ("Run the inner loop concurrently when cores are spare"), which added `execution.multithreading_level` and made the inner loop run several points at once. Its parent is `8c04856`, so `git show 422b00d` or `git diff 8c04856..422b00d` shows exactly the change described here.

This document explains the **bash** side: what gets emitted into the generated `.sh`, why each construct is shaped the way it is, and which bash behaviours forced those shapes.

---

## 1. The problem

A SLURM task gets `cpus-per-task` cores. If the executable only uses some of them, the rest idle for the whole job:

```
cpus-per-task: 16,  executable uses 4 threads

before:   [L1....][L2....][L3....][L4....]      4 cores busy, 12 idle
after:    [L1....][L5....]
          [L2....][L6....]                       16 cores busy
          [L3....][L7....]
          [L4....][L8....]
```

You declare how many cores *one* invocation uses; the orchestrator works out how many fit:

```json
"slurm":     { "cpus-per-task": 16 },
"execution": { "multithreading_level": 4 }      →  16 / 4 = 4 concurrent runs
```

Omitting `multithreading_level` keeps the old strictly-sequential behaviour.

---

## 2. Two levels of code: generator vs. generated

This is the thing to keep straight while reading the source. Almost all the bash discussed below **does not run on your machine** — it is text that Python writes into a file, which SLURM later runs on a compute node.

```
  config.json                                        (you write this)
       │
       ▼
  orchestrator.py  +  mode_builders.py               PYTHON, runs once, on the login node
       │                                             - expands loops
       │                                             - drops already-completed work
       │                                             - pastes bash templates together
       ▼
  submit_inner_<exp>_<timestamp>.sh                  TEXT: a self-contained bash script
       │
       ▼
  sbatch  ──▶  compute node                          BASH, runs the actual physics
```

The bash lives in `mode_builders.py` as four template strings:

| Template | Line | Emitted when | Purpose |
| :--- | :--- | :--- | :--- |
| `RUN_INNER_TASK_TMPL` | `mode_builders.py:34` | always | the worker: runs **one** inner point |
| `SEQUENTIAL_LOOP_TMPL` | `mode_builders.py:90` | no `multithreading_level` | plain `for` loop |
| `PARALLEL_RUNTIME_TMPL` | `mode_builders.py:112` | `multithreading_level` set | slot/buffer/trap machinery |
| `PARALLEL_LOOP_TMPL` | `mode_builders.py:224` | `multithreading_level` set | the dispatcher loop |

`build_inner_driver()` (`mode_builders.py:271`) picks and concatenates them:

```
multithreading_level unset →  RUN_INNER_TASK_TMPL + SEQUENTIAL_LOOP_TMPL
multithreading_level set   →  RUN_INNER_TASK_TMPL + PARALLEL_RUNTIME_TMPL + PARALLEL_LOOP_TMPL
```

Both inner-loop mode and job-array mode call this same function, so the worker exists in exactly one place. Before `422b00d` those two modes each had their own near-identical copy of the loop body.

### Why `@@TOKEN@@` and not f-strings

The rest of the generator builds bash with Python f-strings, which forces every literal bash brace to be doubled (`${{indicator}}`, `$(( ))` → `$(( ))`). That is tolerable for ten lines and a menace for eighty. These templates are therefore plain raw strings with `@@TOKEN@@` placeholders substituted by `.replace()`:

```python
RUN_INNER_TASK_TMPL = r'''
    local indicator="@@INDICATOR_PREFIX@@L${line_no}"     # no brace doubling needed
'''
```

`@@INDICATOR_PREFIX@@` becomes `""` for inner-loop mode and `A${TASK_ID}_` for array mode — that one token is the entire difference between the two modes' workers. A missed brace would only fail when the job starts on the node, long after `sbatch` accepted it, so the generator now also pipes every script through `bash -n` before writing it (`orchestrator.py:304`).

---

## 3. Three emitted pieces

### Piece 1 — the worker, `run_inner_task()`

Runs exactly one inner point. Same function in both sequential and parallel mode; only *how it is called* differs.

```bash
run_inner_task() {
    line_no, inner_args, outer_args, cpu_list, CHECKPOINT_FILE   # 5 positional args

    taskset -cp "$cpu_list" $BASHPID          # pin to this slot's cores
    build exp_args (--save_dir …)
    print_args …

    {
        starttime=…
        eval "<interpreter> <exec> <args> …"  # the actual physics
        EXIT_CODE=$?
        touch "$CHECKPOINT_FILE"  if exit 0
        print duration
    } > >(sed -u "s/^/[${indicator}_out] /") \
     2> >(sed -u "s/^/[${indicator}_err] /" >&2)

    wait                                      # ← MANDATORY, see §4.2
    return "$EXIT_CODE"
}
```

### Piece 2 — the runtime (parallel only)

Computes concurrency, creates the buffer directory, defines helpers, installs traps, slices the CPU mask.

```bash
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
NJOBS=$(( NCPU / MULTITHREADING_LEVEL ))
clamp NJOBS to MAX_PARALLEL, floor at 1
LOG_BUF_DIR="${SLRM_OUTPUT_DIR:-$PWD}/.partial/${SLURM_JOB_ID:-$$}"
```

The `:-$(nproc)` fallback is not cosmetic. `cpus-per-task` is not a required key, and with `SLURM_CPUS_PER_TASK` unset bash arithmetic treats the empty string as `0`, giving `NJOBS=0` — a dispatcher that never launches anything and spins forever.

### Piece 3 — the dispatcher (parallel only)

The `for` loop that hands points to workers, keeps at most `NJOBS` alive, and prints their output.

---

## 4. The bash mechanics, one at a time

These are the parts that look strange. Each one is a response to a specific measured behaviour of bash 5.1.8 on this cluster.

### 4.1 The worker *must* be a function

This exact construct **hangs forever** at the top level of a script:

```bash
{
  echo hi
} > >(sed "s/^/[o] /")
wait                        # ← never returns
```

Verified: `timeout` kills it at rc=124. The identical block placed **inside a function body** works fine and `return` propagates correctly. This is why the worker is a function rather than an inlined block, and why the generated script carries a comment saying so — "simplifying" it back inline would produce a job that hangs until the walltime expires.

### 4.2 Process substitution and the mandatory `wait`

`> >(sed …)` starts `sed` as a **separate asynchronous process**. The shell does not automatically wait for it:

```bash
f(){ { echo hi; } > >(sleep 0.4; sed "s/^/[p] /"); }
f; echo AFTER
#  →  AFTER                    ... and "[p] hi" is LOST entirely
```

Add `wait` and it behaves:

```bash
f(){ { echo hi; } > >(sleep 0.4; sed "s/^/[p] /"); wait; }
f; echo AFTER
#  →  [p] hi
#     AFTER
```

So every worker ends with `wait` before `return`. Without it, output is silently dropped — the worst kind of failure, because the run still writes its checkpoint and looks successful.

Why this is safe in both modes: in the parallel path the worker runs inside a backgrounded subshell, so its `wait` only sees its own two `sed` processes; in the sequential path the parent has no other background jobs. The dispatcher itself never uses process substitution, because its `wait -n` would otherwise reap a `sed` and corrupt the pid bookkeeping.

### 4.3 `sed -u`

GNU sed block-buffers 4 KB when its output is a regular file (which, in parallel mode, it now is). Unbuffered mode matters twice: on a walltime kill you would otherwise lose up to 4 KB from the tail of every in-flight run, and `-u` is what makes a running job's buffer file usefully `tail -f`-able.

### 4.4 Atomic output without any locking

The requirement was that runs never interleave. The mechanism:

```
   worker 1  ──writes──▶  .partial/<jobid>/L1.log  ─┐
   worker 2  ──writes──▶  .partial/<jobid>/L2.log  ─┼─▶  parent cats ONE whole
   worker 3  ──writes──▶  .partial/<jobid>/L3.log  ─┘    file at a time
                                                              │
                                                              ▼
                                                     SLURM .out  (single writer)
```

Workers only ever write to their own private file. The **parent is the only process that writes to the SLURM log**, and it writes one complete buffer per `cat`. Single-writer gives atomicity for free — no `flock`, no lock contention, no chance of a torn line.

`_dump_buffer` also appends a newline if the file does not end with one, so the next block's `[Lx_out]` prefix cannot get glued onto a previous run's unterminated final line.

### 4.5 The sliding window: `wait -n -p`

Bash 5.1 added `-p`, which reports *which* child finished:

```bash
wait -n -p fp      # blocks until any one child exits; sets $fp to its PID, $? to its status
```

That is the whole trick. Two associative arrays map that PID back to what the dispatcher needs:

```bash
declare -A PID_BUF PID_SLOT     # pid → buffer file, pid → slot number
```

`_reap_one` then: waits → looks up the buffer → dumps it → returns the slot to the free list → updates counters.

If a trapped signal interrupts `wait -n`, it returns with `fp` unset; `_reap_one` checks for that and returns 1 rather than indexing the map with an empty key.

### 4.6 Slots and CPU pinning

A "slot" is a concurrency seat, numbered `0 … NJOBS-1`. The free list is a stack; a launch pops one and a reap pushes it back.

Each slot maps to a disjoint slice of the job's own CPU mask:

```
job mask: 0,1,2,3,4,5,6,7        MULTITHREADING_LEVEL=2

slot 0 → 0,1      slot 1 → 2,3      slot 2 → 4,5      slot 3 → 6,7
```

`_slot_cpus` builds that slice and the worker applies it with `taskset -cp … $BASHPID`, which pins the subshell; **children inherit the mask**, so the physics process sees only its own cores (`nproc` inside reports 2, verified).

This is not an optimisation, it is a correctness fix for configs like `dip_strong_scaling/dip_pinned_16.json`, which sets `JULIA_EXCLUSIVE: 1`. Julia then pins its threads to the first *N* cores of whatever mask it is given. Without per-slot masks, all concurrent runs would pin onto the *same* cores while the rest of the node idled — slower than running sequentially, and silently so.

### 4.7 Traps

```bash
trap '_flush_partial' EXIT
trap 'echo "[SIGNAL] …"; _flush_partial; exit 143' TERM
```

`_flush_partial` prints any still-running run's buffer under a `[PARTIAL]` banner. It is guarded by a `_FLUSHED` flag because `exit` inside the TERM handler re-triggers the EXIT handler — without the guard, everything prints twice.

The `[PARTIAL]` banner deliberately contains neither `_out]` nor `Job duration:`, so `--collect`'s log regexes ignore it.

Note this trap is only a *safety net*. Because output is emitted as each run finishes (§5), completed runs are already in the log before any signal arrives.

### 4.8 Keeping checkpoint hashes identical

The riskiest part of the refactor. A run's checkpoint filename contains an MD5 of its full invocation, and Python recomputes those same hashes to decide what to skip (`checkpoint_utils.py`). If the string being hashed had changed by even one space, **every existing `.done` marker in every experiment would have silently become invalid**, re-running months of finished compute.

Before, the two modes hashed different strings:

```bash
mode 1:   "$EXEC_SIG $inner_args_str"
mode 2:   "$EXEC_SIG $outer_args_str $inner_args_str"
```

The unified template has to produce both from one line. Hence:

```bash
"$EXEC_SIG${outer_args_str:+ $outer_args_str} $inner_args_str"
```

`${var:+ $var}` expands to `" $var"` only when `$var` is non-empty, and to nothing when it is empty. Inner-loop mode sets `outer_args_str=""`, so it reduces exactly to the old mode-1 string; array mode has it populated, giving exactly the old mode-2 string. Both were checked by direct `md5sum` comparison, and end-to-end: 33 runs → 33 markers → re-running the script skipped all 33 → re-running the orchestrator reported "All inner loop tasks completed".

---

## 5. The dispatcher, end to end

```
                 ┌─────────────────────────────────────────┐
                 │  for each pending inner point           │
                 └───────────────────┬─────────────────────┘
                                     ▼
                        ┌────────────────────────┐
                        │ compute hash+indicator │
                        │ checkpoint file exists?│
                        └───────┬────────┬───────┘
                            yes │        │ no
                                ▼        ▼
                    "[CHECKPOINT]   ┌───────────────────┐
                     Skipping"      │   NJOBS <= 1 ?    │
                    (no slot used)  └────┬─────────┬────┘
                                     yes │         │ no
                                         ▼         ▼
                          run in parent      ┌──────────────────────┐
                          (streams live,     │ all slots busy?      │
                           old behaviour)    │  → _reap_one()       │◀──┐
                                             └──────────┬───────────┘   │
                                                        ▼               │
                                             ┌──────────────────────┐   │
                                             │ pop slot             │   │
                                             │ launch worker &      │   │
                                             │ stdout+stderr ▶ buf  │   │
                                             │ PID_BUF/PID_SLOT[$!] │   │
                                             └──────────┬───────────┘   │
                                                        └───────────────┘
                                     ┌──────────────────────────────┐
   after the loop:                   │ while workers remain:        │
                                     │   _reap_one()  (drain)       │
                                     └──────────────┬───────────────┘
                                                    ▼
                                        "[SUMMARY] completed=… skipped=… failed=…"


   _reap_one():   wait -n -p fp  ─▶  cat that pid's buffer to the log
                                 ─▶  push its slot back on the free list
                                 ─▶  count ok / failed
```

Two details worth noting:

- The **checkpoint test happens in the dispatcher, before launching**, so an already-finished point never occupies a slot.
- When `NJOBS` computes to 1 (small allocation, or no `multithreading_level`), the worker is called directly in the parent and streams straight to the log — byte-for-byte the old behaviour, no buffering, still `tail -f`-able.

---

## 6. Why output is not in `L1, L2, L3` order

Output blocks appear in **completion order**. This was a deliberate choice (it replaced an earlier input-ordered design) and it buys crash resilience.

```
NJOBS=3, durations differ

 time ──────────────────────────────────────────────▶
 slot 0 │ L1 ================================ │ L4 ====
 slot 1 │ L2 ========= │ L5 =============== │
 slot 2 │ L3 ===== │ L6 ================ │

 log    │         ↑L3      ↑L2       ↑L5   ↑L1   ↑L6 …
                  each block written the moment its run is reaped
```

Had emission been forced into input order, L3 and L2 would have had to sit in their buffers until slow L1 finished — and a walltime kill at that moment would strand several *completed* runs' output in files nobody reads. With completion order, **anything that finished is already in the SLURM log.**

Measured on a deliberate `SIGTERM` mid-run at 3-way concurrency: 3 completed runs present in full with their checkpoint markers, the 3 in-flight runs flushed as `[PARTIAL]`, exit code 143.

Ordering costs nothing elsewhere: `--collect` matches per line with the indicator captured inside each match, so it never depended on order.

Live output for a run still executing is at:

```
<slrm_output_dir>/.partial/<job_id>/<indicator>.log
```

The directory uses `$SLURM_JOB_ID`, **not** the exported `$JOB_ID`. In array mode `JOB_ID` is `SLURM_ARRAY_JOB_ID`, which is identical for every element of the array — sharing one buffer directory across concurrently running array tasks would let their cleanup delete each other's live buffers.

---

## 7. What was deliberately not used

| Option | Why not |
| :--- | :--- |
| **GNU parallel** | Not installed on this cluster and no module provides it (only unrelated `parallel-netcdf`). Vendoring it adds a runtime dependency to a framework whose scripts are otherwise self-contained, and it puts another parsing layer over argument strings that are not shell-quoted. |
| **`xargs -P`** | Available, but provides *no* output grouping at all — measured: it interleaves line by line. Per-run buffers would still be needed, and it gives no per-completion hook, so output could only be dumped after the whole loop. |
| **`flock`** | Would still require buffering (you cannot hold a lock for a run's whole duration), so it adds a lock without removing anything. Having the parent be the sole writer achieves the same guarantee with no lock. |
| **Buffering in a shell variable** | Loses everything on SIGKILL, strips trailing newlines, and charges output volume to the job's memory cgroup. |
| **`shlex.quote` on argument values** | Would change `exec_sig_str`, hence every MD5, hence invalidate every existing `.done` marker. Only ever safe as a deliberate, versioned migration. |

---

## 8. Using it

```json
"slurm":     { "cpus-per-task": 16, "mem": "40G" },
"execution": {
  "multithreading_level": 4,
  "max_parallel_tasks": 3,
  "env_vars": { "JULIA_NUM_THREADS": "$MULTITHREADING_LEVEL" }
}
```

**Set your thread variable yourself.** The generator exports `MULTITHREADING_LEVEL` (`orchestrator.py:203`) but never rewrites `env_vars`. Leaving `JULIA_NUM_THREADS` at `$SLURM_CPUS_PER_TASK` while running N ways concurrently asks for N × the cores and is slower than sequential; `warn_about_concurrency` (`orchestrator.py:266`) prints a warning but does not block.

Guards in `config_schema.py`:

- `multithreading_level` has `ge=1` — `0` is rejected at validation rather than dividing by zero on the node.
- A level larger than `cpus-per-task` is rejected (`config_schema.py:300`): one run would not fit in the allocation.
- `max_parallel_tasks` caps concurrency independently of core count, which is the lever to use when `--mem` binds before cores do. Generation prints the per-run memory share so you can check it.

**Requirement on the executable:** all concurrent runs share one working directory (the script `cd`s to the executable's parent). The executable must therefore be safe to run N times at once from the same directory — anything it writes to a path relative to CWD will collide. Per-run outputs are fine, since `--save_dir` embeds the indicator.

### Verifying a change

```bash
# 1. sequential must be unchanged
python3 orchestrator.py templates/loop_test.json --dryrun

# 2. parallel
python3 orchestrator.py <config with multithreading_level> --dryrun

# 3. every generated script is bash -n checked automatically, and refused
#    above ~120 KB (SLURM's max_script_size is 131072)
```

The checkpoint round trip is the test that matters most: run the script, re-run it (every point should report `[CHECKPOINT] Skipping`), then re-run the orchestrator (it should report nothing left to submit). If the hashing ever drifts, that sequence catches it immediately.
