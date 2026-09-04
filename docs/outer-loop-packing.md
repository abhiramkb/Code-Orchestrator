# Packing the outer loop into fewer array jobs

Implementation notes for the change that added `slurm.num_array_jobs` and the partition time-limit check. Baseline is `227a4fb`; at the time of writing this change is still in the working tree, so `git diff` shows it.

Companion to [parallel-inner-loop.md](parallel-inner-loop.md), which covers `run_inner_task` and the concurrency dispatcher. Everything described there is reused unchanged.

---

## 1. The problem

Mode 2 mapped exactly one outer-loop combination to one SLURM array element. That runs into three separate walls:

| Wall | Value on this cluster |
| :--- | :--- |
| `MaxArraySize` | **1001** — a sweep with more combinations simply cannot be submitted |
| Association submit limits | `maxsubmit` as low as **8** on some `lappi` accounts (also 100, 200) |
| Queue pressure | thousands of tiny elements are unfriendly to the scheduler |

Packing decouples "how many parameter combinations" from "how many jobs":

```
 100 outer combinations, num_array_jobs: 80

 before:  element 0 → combo 0        after:  element 0 → combos 0,1
          element 1 → combo 1                element 1 → combos 2,3
          ...                                ...
          element 99 → combo 99              element 19 → combos 38,39
                                             element 20 → combo 40
          --array=0-99                       ...
          (100 elements)                     element 79 → combo 99
                                             --array=0-79
                                             (80 elements: 20 of size 2, 60 of size 1)
```

---

## 2. The trap this change had to avoid

This is the part worth understanding, because getting it wrong would have corrupted results silently rather than failing loudly.

Before this change, the array element id and the outer combination index were **the same number**. The run indicator was built from the element id:

```bash
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
outer_args_str="${OUTER_ARGS[$TASK_ID]}"
indicator="A${TASK_ID}_L${line_no}"          # ← keyed on the ARRAY ELEMENT
```

That indicator is what names the per-run output directory, via the tracking args (`orchestrator.py`, `SAVE_DIR`):

```
${SAVE_DIR}_${indicator}      e.g.  /scratch/.../1076474_A0_L5
```

Once one element holds several combinations, `A<element>_L<line>` stops being unique:

```
element 0 owns combos 7 and 8
    combo 7, inner line 5  →  indicator A0_L5  →  .../1076474_A0_L5
    combo 8, inner line 5  →  indicator A0_L5  →  .../1076474_A0_L5     ← SAME DIRECTORY
```

Two different physics runs writing into one directory, each overwriting the other's `result.json`. Checkpoint markers would have survived by luck — the marker is `${indicator}_${hash}.done` and the hash includes the outer args, so the filenames still differ — which makes it worse: the campaign would look complete while half the results were destroyed.

**The fix is to key the indicator on the outer combination index instead:**

```bash
for o_idx in "${MY_OUTER_IDS[@]}"; do
    outer_args_str="${OUTER_ARGS[$o_idx]}"
    indicator="A${o_idx}_L${line_no}"        # ← keyed on the COMBINATION
```

Because the two numbers are equal when not packing, **every existing indicator string is unchanged**. That gives a property worth stating explicitly:

> Checkpoints and result directories are independent of `num_array_jobs`. You can complete part of a campaign unpacked, then set `num_array_jobs` and resume, and the finished work is still recognised.

This was verified end to end rather than reasoned about — see §6.

In the generator this is a single substitution token, because the indicator prefix was already a `@@TOKEN@@` placeholder from the previous change:

```python
build_inner_driver(ctx, config, indicator_prefix="A${o_idx}_", wrap_outer_loop=True)
```

---

## 3. Which combinations go in which element

`group_outer_tasks` (`mode_builders.py:361`) does the mapping. Two rules matter.

**Rule 1 — split the *pending* list, not the index space.** `get_pending_array_task_ids` already returns only combinations with work left. Partitioning the index space arithmetically would produce lopsided elements on resume:

```
pending = [0, 5, 99]      (997 combinations already finished)

index-space blocks of 10:   element 0 → "combos 0..9"  (only 0 and 5 are real)
                            element 9 → "combos 90..99" (only 99 is real)
pending-list chunks:        element 0 → 0, 5, 99        (all real)
```

**Rule 2 — near-equal groups.** With `N` pending and `G` requested:

```
G        = min(num_array_jobs, N)        # clamp, else empty elements get submitted
base, r  = divmod(N, G)
group i  = base + 1 combinations   if i < r
           base                    otherwise
```

Sizes therefore differ by at most one. That matters for more than tidiness — see §5 on the shared time limit.

Checked against real inputs:

| N pending | `num_array_jobs` | elements | group sizes |
| ---: | ---: | ---: | :--- |
| 100 | 80 | 80 | 20×2 + 60×1 |
| 100 | 8 | 8 | 4×13 + 4×12 |
| 5 | 3 | 3 | 2, 2, 1 |
| 5 | 80 | **5** | 1 each (clamped) |
| 1000 | 50 | 50 | 20 each |

When `num_array_jobs` is unset, each pending combination becomes its own group *keyed by its original index*, which reproduces the historical sparse `--array=0-2,5,8` exactly.

---

## 4. What the generated script gained

### Membership as data, not arithmetic

Python already knows the mapping, so it is emitted rather than recomputed in bash. A bash **sparse indexed array** expresses both modes with one code path:

```bash
# packed                                  # unpacked (element id == combination index)
TASK_MEMBERS=(                            TASK_MEMBERS=(
    [0]="0 1"                                 [0]="0"
    [1]="2 3"                                 [5]="5"
    [2]="4"                                   [99]="99"
)                                         )

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
read -ra MY_OUTER_IDS <<< "${TASK_MEMBERS[$TASK_ID]}"
```

Arithmetic partitioning (`start = TASK_ID * P`) only works on a dense index space, which is exactly what a resumed run does not have. The array range itself still comes from the existing `format_slurm_array_range` — handed `range(G)` when packing (dense `0-79%20`) or the pending list when not (sparse, as before).

### The nesting, and where the drain goes

The inner-loop body is now wrapped in a loop over the element's combinations:

```
run_inner_task() { ... }                      ← unchanged

[concurrency runtime: slots, buffers, traps]  ← unchanged

for o_idx in "${MY_OUTER_IDS[@]}"; do         ← NEW wrapper
    outer_args_str="${OUTER_ARGS[$o_idx]}"
    echo "---------- outer combination $o_idx: ... ----------"

    for idx in "${!INNER_ARGS[@]}"; do        ← existing inner body, unchanged
        ... checkpoint check ...
        ... dispatch into a free slot ...
    done
done                                          ← wrapper closes HERE

while [ "$_running" -gt 0 ]; do _reap_one; done   ← drain, OUTSIDE the wrapper
echo "[SUMMARY] ..."
```

**The drain placement is the performance-relevant detail.** Had it stayed inside the wrapper, the slot pool would empty at every combination boundary:

```
drain inside (wrong)                    drain outside (implemented)
 slot 0 │ c0 ══════╗          ╔═ c1      slot 0 │ c0 ══════╤═ c1 ═══════
 slot 1 │ c0 ════╗ ║          ║          slot 1 │ c0 ════╤═ c1 ═════════
 slot 2 │ c0 ══╗ ║ ║          ║          slot 2 │ c0 ══╤═ c1 ═══════════
                ╰─╨─╨ idle ───╯                        ╰ no idle gap
```

With `num_array_jobs: 8` on a 100-combination sweep there are 12 such boundaries per element, so the idle time is not academic.

### The template split

Achieving that meant splitting the two loop templates into a body and a tail, so `build_inner_driver` can assemble *body-inside-wrapper* followed by *tail*:

| Template | Line | Contents |
| :--- | :--- | :--- |
| `SEQUENTIAL_LOOP_TMPL` | `mode_builders.py:105` | the `for` body |
| `SEQUENTIAL_TAIL_TMPL` | `mode_builders.py:127` | `wait` |
| `PARALLEL_LOOP_TMPL` | `mode_builders.py:244` | the dispatch body |
| `PARALLEL_TAIL_TMPL` | `mode_builders.py:286` | drain + `[SUMMARY]` + `wait` |
| `OUTER_WRAPPER_HEAD/TAIL_TMPL` | `mode_builders.py:93`, `100` | the wrapper |

Assembly is a small closure in `build_inner_driver` (`mode_builders.py:342`) using `textwrap.indent`, so the wrapped body is indented correctly and stays readable in the emitted script. Mode 1 passes `wrap_outer_loop=False` and an empty prefix, so it never references `o_idx` and is completely unaffected.

---

## 5. The shared time limit

One script means one `#SBATCH --time`, applying to **every** element regardless of how many combinations it holds. So it must cover the *largest* group.

This is why near-equal groups matter: with sizes differing by at most one, an element holding the minimum over-requests by at most one combination's worth of runtime. Any lumpier split would waste more, and over-requesting also makes a job a worse backfill candidate.

Deliberately **not** implemented: a `time_per_outer` knob that computes `--time` for you. Sizing it stays your call. What the generator does instead is report the arithmetic it cannot do for you:

```
[INFO] Packing 100 outer combination(s) into 8 array element(s); up to 13 per element.
[INFO] '--time' applies to every element, so it must cover the largest
       (13 combination(s) run back to back).
```

---

## 6. The partition time-limit check

Modelled on the existing executable `--help` check (`validate_script_args`, `orchestrator.py:539`): a violation is a hard error, not a warning, since `sbatch` would reject it anyway — only later, after you have waited.

`validate_partition_time_limit` (`orchestrator.py:331`) queries `scontrol show partition <name>` and compares `MaxTime` to the requested `time`:

| Situation | Behaviour |
| :--- | :--- |
| requested > `MaxTime` | `[ERROR]` naming both values, exit 1 |
| partition does not exist | `[ERROR]`, exit 1 — catches a typo'd partition name |
| `MaxTime=UNLIMITED` | pass |
| `scontrol` not installed | skip silently (so `--dryrun` works off-cluster) |
| `--notimecheck` | skip |

Supporting pieces: `parse_slurm_time_seconds` (`config_schema.py:154`) handles every format sbatch accepts — `MM`, `MM:SS`, `HH:MM:SS`, `D-HH`, `D-HH:MM`, `D-HH:MM:SS` — and `format_seconds_as_slurm_time` (`orchestrator.py:366`) renders the message back in `D-HH:MM:SS`.

**It runs at generation, not at submission.** The argument check lives in `submit_slurm_script` because it is expensive, but the documented default workflow generates a script and then tells you to `sbatch` it by hand, so a submit-time-only check would miss most real usage.

This is not hypothetical for packed configs. `gpumedium` caps at `1-12:00:00`; packing two `23:59:00` combinations onto it gives `1-23:58:00`, which the check now rejects up front. Partition caps here: `test`/`gputest` 15 min, `medium`/`large`/`hugemem`/`gpumedium` 1-12, `small` 3-00, `longrun`/`hugemem_longrun` 10-00.

---

## 7. Verification

The checkpoint-compatibility claim in §2 is the one that had to be proven, not argued:

```
1. run 5 combos × 3 points UNPACKED     → markers A0_L1 … A4_L3   (15)
2. regenerate WITH num_array_jobs: 3    → "All job array tasks are complete"
3. wipe, run the same sweep PACKED      → markers A0_L1 … A4_L3   (15, identical set)
4. inspect the 15 save_dir values        → 15 distinct directories, no collision
5. confirm per-combination arguments     → A0→Csq 100 … A4→Csq 500, no bleed
```

Step 2 is the important one: markers written by unpacked elements are recognised by a packed configuration.

Other checks: the unpacked path still emits `--array=0-99` with an unchanged hash line for a real 100-combination config; concurrency plus packing ran 2 combinations × 3 points 3-way concurrent in one element (`[SUMMARY] completed=6`); all 45 repository configs generate cleanly; every script passes the automatic `bash -n`.

---

## 8. What did not change

Worth being explicit, since the surface area sounds larger than it is:

- **`run_inner_task`** — untouched. It already took `outer_args_str` as a parameter.
- **The concurrency dispatcher** — untouched. Slots, buffer files, `wait -n -p`, traps, CPU pinning all behave identically; the pool is simply shared across combination boundaries now.
- **Checkpoint hashing** — untouched, including the `${outer_args_str:+ $outer_args_str}` trick that keeps mode 1 and mode 2 hashing identical strings.
- **`checkpoint_utils.py`** — untouched. `get_pending_array_task_ids` still returns pending combination indices; grouping happens after it.
- **Mode 0 and mode 1** — unaffected.

## 9. Using it

```json
"slurm": {
  "partition": "small",
  "time": "2-00:00:00",
  "num_array_jobs": 80,
  "max_concurrent_tasks": 20
}
```

`num_array_jobs` sets the array's **width**, `max_concurrent_tasks` its **throttle** (`--array=0-79%20`). Both are consumed by the orchestrator rather than emitted as `#SBATCH` directives.

Gotchas:

- **Size `time` for the largest group.** The generator prints it; the partition check catches an over-limit result.
- **Packing is computed over pending work**, so a resumed run repacks whatever is left rather than reserving elements for finished combinations.
- **Failure blast radius grows.** A node failure now affects every combination in that element rather than one — though checkpointing means only in-flight inner points are lost, and a resume picks up the rest.
