# Orchestrator — Quick Start

**Goal of this page:** get a parameter scan running on SLURM without reading the full manual.
For every option and the details of how it works, see [README.md](README.md).

---

## What it does

You write **one YAML file** that says: what program to run, what your `#SBATCH` header
should look like, and which parameters to loop over. The orchestrator expands the parameter
lists and writes a ready-to-go `sbatch` script (`submit_*.sh`), optionally submitting it.

Two useful facts up front:

- **Re-running the same command resumes.** Runs that already finished are skipped, so
  after a walltime kill you just submit the same config again.
- **JSON and YAML both work.** Same fields either way. Examples here use YAML.

---

## The only commands you need

```bash
cd /projappl/lappi/abhiram/Code-Orchestrator
CFG="yaml_nlo_diff/dip_medianbk_mvgam,balsd_HERA.yaml"

python3 orchestrator.py "$CFG"            # write the script, don't submit
python3 orchestrator.py "$CFG" --submit   # write it and sbatch it
python3 orchestrator.py "$CFG" --collect  # gather finished results into results.db
```

| Flag | Use it when |
| :--- | :--- |
| `--submit` | You want it queued now. Without this you just get the `.sh` to inspect or `sbatch` yourself. |
| `--dryrun` | You want to see the generated script without touching SLURM (also skips "does this file exist" checks). |
| `--noargcheck` | Startup complains your executable doesn't support a flag, but you know it does. |
| `--notimecheck` | You don't want `time` validated against the partition's `MaxTime`. |
| `--collect` | Jobs are done and you want `results.db`. |
| `--collect-job 12345678` | Same, one job only → `results_12345678.db`. |

---

## Start from an example — don't write one from scratch

Pick the row that matches what you want and copy that file:

| I want to… | Copy this | You get |
| :--- | :--- | :--- |
| Run one dipole over a list of kinematic points | `yaml_nlo_diff/dip_medianbk_mvgam,balsd_HERA.yaml` | **One job**, one run per line of the points file, done in sequence |
| Scan 100 dipole replicas × all kinematic points | `yaml_nlo_diff/dip_kcbk_allbks_mvgam,balsd_HERA.yaml` | **A job array**, one task per replica, each task sweeping every point |
| Do a single run, one set of numbers | any `*_medianbk_*` file: delete `inner_loop`, add `loopQ: false`, put `Q`/`beta`/`x` into `args` | One job, one run |

`yaml_nlo_diff/` has 32 ready configs covering dip / dipLO / diptrip / trip against the various
dipole fits. The closest one is usually a 3-line edit away from what you need.

---

## Example 1 — one job, sweeping a list of points

This is `yaml_nlo_diff/dip_medianbk_mvgam,balsd_HERA.yaml` with comments added:

```yaml
slurm:                                # your normal #SBATCH header, one key per option
  account: lappi
  job-name: dip_HERA
  partition: small                    # required
  nodes: 1
  ntasks: 1
  time: "23:59:00"                    # required. quote it, or YAML reads it as a sexagesimal number
  cpus-per-task: 16
  mem: 10G
  output: slurm_%j.log

execution:                            # what to run
  interpreter: julia                  # omit if the executable runs on its own
  executable: /projappl/lappi/abhiram/NLO_Diffraction_dip-dip/sf_nlo_sdaw_dip_L.jl
  modules:                            # module load ... before running
    - openmpi
    - julia/1.11.9
    - julia-mpi
  env_vars:
    JULIA_NUM_THREADS: $SLURM_CPUS_PER_TASK

experiment:                           # where results go
  result_database_path: /scratch/lappi/abhiram/dip_database
  experiment_name: median_mvgam_balsd # subdirectory name, also used in the script filename
  slrm_output_dir: SLURM_OUTPUT
  tracking_args:                      # extra args handed to your code per run
    save_dir: "{save_dir}_{__indicator}"
    json: "{save_dir}_{__indicator}/result.json"

inner_loop:                           # ONE RUN PER LINE of this file
  file_path: /projappl/lappi/abhiram/trip/HERA_kinematic_points_highQ_smallx.txt
  delimiter: " "
  arg_names:                          # column 0 -> --Q, column 1 -> --beta, column 2 -> --x
    - --Q
    - beta                            # bare names get "--" added automatically
    - --x

args:                                 # fixed args added to every single run
  dipole_path: /projappl/lappi/abhiram/nlobk-nlodisft/data/mvgam,balsd/median_bk.dat
  Csq: 915
  xmax: 160.0
  neval: 400000000.0
```

That's the whole mental model: **`inner_loop` varies, `args` stays fixed.**

Don't have a data file? Sweep values inline instead:

```yaml
inner_loop:
  type: explicit
  arg_name: Csq
  values: [100, 315, 915]
```

---

## Example 2 — a scan (job array)

Take Example 1 and add an `outer_loops` block. That single addition turns the job into an
array: **one array task per outer combination**, and each task runs the whole inner loop.

```yaml
outer_loops:
  - type: tabular_file                # required
    file_path: /projappl/lappi/abhiram/bayesian_alldata/KCBK_Balsd_MVgamma/posteriorsamples.dat
    delimiter: ","
    comment_prefix: "#"
    skip_blank_lines: true
    args:
      - arg_name: dipole_path
        column: 0                     # 0-based. column 0 holds the replica index: 0, 1, 2, ...
        template: /projappl/lappi/abhiram/bayesian_alldata/KCBK_Balsd_MVgamma/bks/{val}.dat
      - arg_name: Csq
        column: 3
```

100 replicas × 51 points = 100 array tasks of 51 runs each. Remember to drop `dipole_path`
and `Csq` from `args` — they come from the table now.

**File has more than one kind of metadata line?** `comment_prefix` also takes a list:

```yaml
    comment_prefix: ["#", "*", "!"]     # skip lines starting with any of these
```

**Column specs, in short:**

| Field | Meaning |
| :--- | :--- |
| `column` | Which column to read, counting from **0**. |
| `source` | `column` (the default) reads a column. `row_index` uses the row's own position instead — see below. |
| `index_offset` | With `source: row_index`, added to the index. Use `1` if your files start at `1.dat`. |
| `template` | Build a string from the value, e.g. `bks/{val}.dat`. |
| `transform` | Math on the value first, e.g. `"10^x"`, `"log10(x)"`, `"sqrt(x)"`. |

> **The one gotcha worth checking:** whether your posterior file stores `C^2` or `log10 C^2`.
> Look at the `### Columns:` header line. The KCBK Balsd fit stores plain `C^2` → **no**
> `transform`. The KCBK pd fit and the NLO fits store `log10 C^2` → `transform: "10^x"`.
> Getting this wrong silently produces nonsense, and nothing will warn you.

**No replica index in your file?** The example above relies on column 0 holding `0, 1, 2, …`.
Plenty of posterior files don't have that column, even though their rows still line up with
`bks/0.dat`, `bks/1.dat`, … Let the orchestrator count instead:

```yaml
      - arg_name: dipole_path
        source: row_index             # 0, 1, 2, ... in file order — no column needed
        template: /projappl/lappi/abhiram/bayesian_alldata/LO_MVe/bks/{val}.dat
```

> The index counts **data rows, not file lines**: a `#` header doesn't shift it, so the first
> row of numbers is `0` even when it sits on line 5. Add `index_offset: 1` if your files start
> at `1.dat`, or write `{val:03d}` in the template for zero-padded names like `007.dat`.

This works the same in `inner_loop` args — a working example is
`bk_evol_tf/compare_evolution_carlisle_lo_replicas.yaml`.

Other outer-loop types, if you need them: `explicit` (`arg_name` + `values`) and
`range` (`arg_name`, `start`, `stop`, `step`; `stop` is exclusive).

---

## Making a big scan fit, and run faster

Four knobs. Set them in the config you copied.

| Knob | Where | What it does |
| :--- | :--- | :--- |
| `max_concurrent_tasks: 25` | `slurm` | Throttle: at most 25 array tasks running at once (`--array=0-99%25`). |
| `num_array_jobs: 25` | `slurm` | Pack the 100 combinations into 25 array elements (4 each). Use when the scan exceeds `MaxArraySize` or your submission limit. |
| `multithreading_level: 16` | `execution` | One run uses 16 cores → with `cpus-per-task: 192`, 12 inner points run at once inside each task. |
| `max_parallel_tasks: 8` | `execution` | Hard ceiling on that concurrency, if memory is the real limit. |

> **If you set `multithreading_level`, fix your thread variable too:**
> use `JULIA_NUM_THREADS: $MULTITHREADING_LEVEL`, **not** `$SLURM_CPUS_PER_TASK`. Leaving it
> at the full allocation while running 12 ways concurrently is *slower* than not doing it at all.
> You get a warning, not an error.

> **If you set `num_array_jobs`, raise `time`.** Each element now runs several combinations
> back to back, and `time` applies per element. The generator prints the largest group size —
> size the walltime for that.

A current, working example of both: `yaml_nlo_diff/dip_kcbk_allbks_mvgam,pd_HERA.yaml`.

---

## Where things land

```text
<result_database_path>/<experiment_name>/
├── SLURM_OUTPUT/            # job logs (whatever you called slrm_output_dir)
├── checkpoints/             # "this run finished" markers
├── <job_id>_<indicator>/    # one output directory per run ($SAVE_DIR)
└── results.db               # after --collect
```

Log lines are tagged by run, so an array log stays readable:

```text
[A0_L1_out] Job duration: 2.104 seconds      # array task 0, inner line 1
[CHECKPOINT] Skipping A0_L2 - already completed.
[A0_L3_err] Warning: Convergence threshold near limit.
```

Want to watch a run that's still going? Tail `SLURM_OUTPUT/.partial/<job_id>/<indicator>.log`.

---

## Resume, redo, restart

| Situation | Do this |
| :--- | :--- |
| Job hit the walltime, half the points are done | Re-run the same command. Only the unfinished points get submitted. |
| One point needs redoing | Delete its marker in `checkpoints/`. |
| Redo the whole campaign | Delete the `checkpoints/` directory. |
| Changed an argument value | Nothing to do — markers are tied to the exact arguments, so changed runs re-run automatically. |

---

## When it complains

| Message | What it means |
| :--- | :--- |
| `Nothing to submit ... all tasks already checkpointed` | It's all done. Delete markers in `checkpoints/` if you meant to redo it. |
| `Argument validation failed! Offending args: ...` | Your code doesn't advertise that flag in `--help`. Fix the name, or pass `--noargcheck`. |
| Time-limit / partition error | `time` exceeds the partition's `MaxTime`, or the partition name is wrong. Fix it, or pass `--notimecheck`. |
| A path error at startup | A `file_path`/`executable` doesn't exist. Paths are checked before anything is written; `--dryrun` skips the check. |
| `'sbatch' command not found` | You're not on a cluster node. Generate without `--submit`. |

---

## Full reference

[README.md](README.md) — every config field, plus `docs/outer-loop-packing.md` and
`docs/parallel-inner-loop.md` for the two features above in depth.
