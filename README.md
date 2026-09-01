# SLURM Execution Orchestrator

An automated framework for generating and submitting SLURM batch scripts for scientific compute codes, supporting single runs, 1D parameter loops, and multi-dimensional outer/inner parameter scans with resumable checkpointing.

Code conceptualized by me (Abhiram Kaushik) but implemented using the Gemini LLM. Permalink to the chat [here](https://share.gemini.google/WEDBVBqp6mVB)

---

## Overview

`orchestrator.py` reads a JSON or YAML configuration file, validates it against a Pydantic schema, expands the parameter space **in Python**, and writes a self-contained Bash submission script.

Parameter expansion happens at generation time: every loop is unrolled into literal argument strings embedded in the generated script, so the script does not read data files at runtime. Work that has already completed is filtered out before the script is written, so a resumed run only submits what is left.

## Command-Line Usage

```bash
# Use the default configuration file (config.json)
python3 orchestrator.py

# One or more explicit config files
python3 orchestrator.py nlo_diff/dip_medianbk_mv,balsd_HERA.json
python3 orchestrator.py config_a.json config_b.yaml

# Generate and submit in one step
python3 orchestrator.py my_config.json --submit

# Generate without touching SLURM, and skip file-existence checks
python3 orchestrator.py my_config.json --dryrun

# Collect finished results into a SQLite database
python3 orchestrator.py my_config.json --collect
python3 orchestrator.py my_config.json --collect-job 12345678
```

| Flag | Description |
| :--- | :--- |
| `--submit` | Submit the generated script with `sbatch`. |
| `--dryrun` | Generate the script only; also relaxes file-existence validation. |
| `--noargcheck` | Skip probing the executable's `--help` to verify that every configured flag is supported. |
| `--collect` | Parse SLURM logs and write all runs into `<exp_dir>/results.db`. |
| `--collect-job <id>` | Same, restricted to one job ID, written to `<exp_dir>/results_<id>.db`. |

## Execution Modes

The mode is derived from the config; `loopQ` may set it explicitly, otherwise it is inferred from which loop sections are present.

| Mode | Condition | Output | Log prefix |
| :--- | :--- | :--- | :--- |
| **Single run** | `loopQ: false`, or no loop sections | `submit_single_<exp>_<timestamp>.sh` | `[SINGLE_out]` |
| **Inner loop only** | `inner_loop` present, no `outer_loops` | `submit_inner_<exp>_<timestamp>.sh` | `[L1_out]`, `[L2_out]`, … |
| **Job array** | `inner_loop` and `outer_loops` present | `submit_array_<exp>_<timestamp>.sh` | `[A0_L1_out]`, `[A0_L2_out]`, … |

In job array mode the outer combinations form the SLURM array (one task per Cartesian product entry, indexed by `$SLURM_ARRAY_TASK_ID`), and each task iterates the full inner loop sequentially.

---

## Configuration Schema

### Top level

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `execution` | Object | **Yes** | Interpreter, executable, flags, modules, environment variables. |
| `slurm` | Object | **Yes** | Translated into `#SBATCH --<key>=<value>` directives. |
| `loopQ` | Boolean | Optional | Force single-run (`false`) or loop (`true`) mode. Inferred when omitted. |
| `inner_loop` | Object | Conditional | Required in loop mode. The sequential sweep run inside one SLURM task. |
| `outer_loops` | Array | Optional | Turns the run into a SLURM job array over the Cartesian product of these blocks. |
| `experiment` | Object | Optional | Result/checkpoint directory layout and tracking arguments. |
| `args` | Object | Optional | Fixed CLI key-value pairs passed to every invocation. |

### `execution`

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `executable` | String | **required** | Path to the program or script. Its parent directory becomes the working directory. |
| `interpreter` | String | `null` | Command that runs the executable, e.g. `julia`, `python3`. Omit for a directly executable file. Also accepted under its legacy name `language`; prefer `interpreter` in new configs. |
| `flags` | Array | `[]` | Interpreter flags placed before the executable, e.g. `["-u"]`. |
| `modules` | Array | `[]` | `module load` calls emitted before execution. |
| `env_vars` | Object | `{}` | Exported before execution. Values may reference SLURM variables, e.g. `"$SLURM_CPUS_PER_TASK"`. |

### `slurm`

Every key is emitted verbatim as `#SBATCH --<key>=<value>`, so any SBATCH option is accepted (`account`, `mem`, `cpus-per-task`, `ntasks`, …). `partition` and `time` are required; `job-name` defaults to `orchestrator`, `nodes` to `1`, `output` to `slurm_%j.log`.

`max_concurrent_tasks` is consumed by the orchestrator rather than emitted: it becomes the `%N` throttle on the array directive (e.g. `#SBATCH --array=0-7%2`).

### `inner_loop`

Two forms, selected by `type`. A dict with no `type` is treated as `tabular_file` for backward compatibility.

**`tabular_file`** — one run per row of a data file:

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file_path` | Path | `null` | Data file to read. |
| `delimiter` | String | `" "` | Column separator; whitespace splitting when a single space. |
| `comment_prefix` | String | `"#"` | Lines starting with this are skipped. |
| `skip_blank_lines` | Boolean | `true` | Skip empty lines. |
| `arg_names` | Array | `null` | Shorthand: maps columns positionally to argument names. |
| `args` | Array | `[]` | Explicit column specs; use instead of `arg_names` when you need `template` or `transform`. |

**`explicit`** — one run per listed value:

| Field | Type | Description |
| :--- | :--- | :--- |
| `arg_name` | String | Argument name. |
| `values` | Array | Values to sweep. |

### `outer_loops`

An array of blocks; the run expands over their Cartesian product. Each block **must** declare `type`:

- **`explicit`** — `arg_name` plus `values`.
- **`range`** — `arg_name`, `start`, `stop`, `step` (default `1.0`); `stop` is exclusive.
- **`tabular_file`** — `file_path` plus `args` column specs (also `delimiter`, `comment_prefix`, `skip_blank_lines`).

### Column specs (`args` entries)

| Field | Type | Description |
| :--- | :--- | :--- |
| `arg_name` | String | Argument name. Bare names gain `--` (or `-` if single character); names already starting with `-` are passed through unchanged. |
| `column` | Integer | Zero-based column index. |
| `transform` | String | Optional math expression applied to the value, e.g. `"10^x"`, `"log10(x)"`, `"sqrt(x)"`. `x` and `val` both refer to the raw value; `^` means exponentiation. Standard `math` functions plus SciPy special functions (Bessel, gamma, …) are available when SciPy is installed. |
| `template` | String | Optional string template applied after `transform`, e.g. `"/data/bks/{val}.dat"`. |

### `experiment`

| Field | Type | Description |
| :--- | :--- | :--- |
| `result_database_path` | String | Root directory for results. |
| `experiment_name` | String | Subdirectory under the root; also used in generated script names. |
| `slrm_output_dir` | String | Subdirectory for SLURM logs. |
| `tracking_args` | Object | Extra CLI arguments injected per run, supporting the placeholders `{result_database_path}`, `{experiment_name}`, `{slrm_output_dir}`, `{save_dir}` and `{__indicator}`. |

This yields the layout:

```text
<result_database_path>/<experiment_name>/
├── <slrm_output_dir>/       # SLURM logs
├── checkpoints/             # completion markers
├── <job_id>_<indicator>/    # per-run output directories ($SAVE_DIR)
└── results.db               # written by --collect
```

Without an `experiment` block, checkpoints go to `./checkpoints` relative to the submission directory and no result directories are created.

---

## Examples

### Case 1: Job array over posterior samples

Outer loop reads dipole fit parameters from a table (one array task per row); each task sweeps every kinematic point in the inner file.

```json
{
  "slurm": {
    "account": "lappi",
    "job-name": "dip_HERA",
    "partition": "small",
    "nodes": 1,
    "ntasks": 1,
    "time": "1-12:59:00",
    "cpus-per-task": 16,
    "mem": "40G",
    "output": "slurm_%j.log",
    "max_concurrent_tasks": 20
  },
  "execution": {
    "interpreter": "julia",
    "executable": "/projappl/lappi/abhiram/NLO_Diffraction_dip-dip/sf_nlo_sdaw_dip_L.jl",
    "modules": ["openmpi", "julia/1.11.9", "julia-mpi"],
    "env_vars": {
      "JULIA_NUM_THREADS": "$SLURM_CPUS_PER_TASK"
    }
  },
  "experiment": {
    "result_database_path": "/scratch/lappi/abhiram/dip_database",
    "experiment_name": "allbks_mvgam_balsd",
    "slrm_output_dir": "SLURM_OUTPUT",
    "tracking_args": {
      "save_dir": "{save_dir}_{__indicator}",
      "json": "{save_dir}_{__indicator}/result.json"
    }
  },
  "inner_loop": {
    "file_path": "/projappl/lappi/abhiram/trip/HERA_kinematic_points_smallx.txt",
    "delimiter": " ",
    "arg_names": ["--Q", "beta", "--x"]
  },
  "outer_loops": [
    {
      "type": "tabular_file",
      "file_path": "/projappl/lappi/abhiram/nlobk-nlodisft/data/mvgam,balsd/posteriorsamples.dat",
      "delimiter": ",",
      "comment_prefix": "#",
      "skip_blank_lines": true,
      "args": [
        {
          "arg_name": "dipole_path",
          "column": 0,
          "template": "/projappl/lappi/abhiram/nlobk-nlodisft/data/mvgam,balsd/bks/{val}.dat"
        },
        {
          "arg_name": "Csq",
          "column": 3,
          "transform": "10^x"
        }
      ]
    }
  ],
  "args": {
    "xmax": 160.0,
    "neval": 4e8
  }
}
```

With 100 posterior samples and 51 kinematic points this emits `#SBATCH --array=0-99%20`, each task running 51 sequential inner evaluations.

### Case 2: Inner loop only

```json
{
  "slurm": {
    "account": "lappi",
    "job-name": "dip_median",
    "partition": "small",
    "nodes": 1,
    "time": "23:59:00",
    "cpus-per-task": 16,
    "output": "slurm_%j.log"
  },
  "execution": {
    "interpreter": "julia",
    "executable": "/projappl/lappi/abhiram/NLO_Diffraction_dip-dip/sf_nlo_sdaw_dip_L.jl",
    "modules": ["openmpi", "julia/1.11.9"],
    "env_vars": {
      "JULIA_NUM_THREADS": "$SLURM_CPUS_PER_TASK"
    }
  },
  "experiment": {
    "result_database_path": "/scratch/lappi/abhiram/dip_database",
    "experiment_name": "median_mv_balsd",
    "slrm_output_dir": "SLURM_OUTPUT"
  },
  "inner_loop": {
    "file_path": "/projappl/lappi/abhiram/trip/HERA_kinematic_points_smallx.txt",
    "delimiter": " ",
    "arg_names": ["--Q", "beta", "--x"]
  },
  "args": {
    "dipole_path": "/projappl/lappi/abhiram/nlobk-nlodisft/data/mvgam,balsd/median_bk.dat",
    "Csq": 915,
    "xmax": 160.0
  }
}
```

An explicit sweep works the same way and needs no data file:

```json
{
  "inner_loop": {
    "type": "explicit",
    "arg_name": "Csq",
    "values": [100, 315, 915]
  }
}
```

### Case 3: Single run

```json
{
  "loopQ": false,
  "slurm": {
    "account": "lappi",
    "job-name": "single_dip",
    "partition": "test",
    "nodes": 1,
    "ntasks": 1,
    "time": "00:15:00",
    "cpus-per-task": 16,
    "output": "slurm_%j.log"
  },
  "execution": {
    "interpreter": "julia",
    "executable": "/projappl/lappi/abhiram/NLO_Diffraction_dip-dip/sf_nlo_sdaw_dip_T.jl",
    "modules": ["openmpi", "julia/1.11.9", "julia-mpi"],
    "env_vars": {
      "JULIA_NUM_THREADS": "$SLURM_CPUS_PER_TASK"
    }
  },
  "experiment": {
    "result_database_path": "/projappl/lappi/abhiram/dip_database",
    "experiment_name": "mvgam_balsd",
    "slrm_output_dir": "SLURM_OUTPUT"
  },
  "args": {
    "dipole_path": "/projappl/lappi/abhiram/nlobk-nlodisft/data/mvgam,balsd/median_bk.dat",
    "Csq": 915,
    "Q": 1.5811388300841898,
    "beta": 0.18,
    "x": 0.0009,
    "xmax": 160.0,
    "neval": 1e8
  }
}
```

---

## Checkpointing and Resuming

Every individual run is identified by an MD5 hash of its full invocation — interpreter, flags, executable, fixed arguments and loop arguments — so a checkpoint is only reused when the run would be genuinely identical. On success the script touches a marker file:

| Mode | Marker |
| :--- | :--- |
| Single run | `SINGLE_<hash>.done` |
| Inner loop | `L<line>_<hash>.done` |
| Job array | `A<task>_L<line>_<hash>.done` |

`<line>` is the position in the **full** inner loop, so indices remain stable no matter how much of the sweep has already finished.

Completion is checked twice, for different reasons:

1. **At generation time (Python).** Finished work is excluded before the script is written. An inner-loop script contains only the outstanding points; a job array emits a sparse range covering only outer tasks with work left, e.g. `#SBATCH --array=0,5,10,75-99%4`. If nothing is outstanding no script is produced and the run reports `Nothing to submit`.
2. **At run time (Bash).** The script re-checks each marker before executing. This covers a script that is submitted twice or requeued by SLURM after a node failure or time limit.

To resume an interrupted campaign, simply re-run the orchestrator on the same config: completed points are skipped and only the remainder is submitted.

Deleting a marker forces the corresponding run to repeat; deleting the `checkpoints/` directory restarts the whole campaign.

---

## Result Collection

`--collect` scans the SLURM logs under `<result_database_path>/<experiment_name>/<slrm_output_dir>`, pairs each run with the `result*.json` file in its output directory, and writes a flattened row per run into `results.db`. Runs skipped via checkpoint are recorded with a `NULL` duration. Use `--collect-job <id>` to collect a single job into `results_<id>.db`.

> The collector currently assumes the NLO Diffraction project's output layout.

---

## Log Output & Stream Tagging

Generated scripts print a run legend, then tag every stream line with its run indicator and report nanosecond-precision timing:

```text
======================= SLURM ARRAY RUN LEGEND =======================
Array Task ID: 0
Working Directory: /projappl/lappi/abhiram/NLO_Diffraction_dip-dip
Exec Path: /projappl/lappi/abhiram/NLO_Diffraction_dip-dip/sf_nlo_sdaw_dip_L.jl
Fixed Args: --xmax 160.0 --neval 4e8
Outer Loop Combo: dipole_path=/data/bks/sample0.dat, Csq=915
Outer Flags: --dipole_path /data/bks/sample0.dat --Csq 915
Inner Args: --Q, --x, beta
Executing Subtasks: 40 / 100 Outer Runs
======================================================================
[A0_L1_out] Job started at: Fri Aug  7 14:00:00 EEST 2026
[A0_L1_out] Job finished at: Fri Aug  7 14:00:02 EEST 2026
[A0_L1_out] Job duration: 2.104 seconds
[CHECKPOINT] Skipping A0_L2 - already completed.
[A0_L3_err] Warning: Convergence threshold near limit.
[A0_L3_out] Job duration: 1.980 seconds
```

---

## Notes on Removed Behaviour

Earlier versions allowed an outer loop block to select a different inner-loop data file per value via an `inner_files` mapping. Inner loops are now evaluated once and shared across all outer combinations, so `inner_files` is rejected with an explicit error rather than silently ignored. Use a separate config per inner file if they need to differ.
