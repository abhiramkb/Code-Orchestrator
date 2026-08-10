# SLURM Execution Orchestrator

An automated framework for generating and submitting SLURM batch scripts for scientific compute codes, supporting single runs, 1D parameter loops, and multi-dimensional outer/inner parameter scans.

Code conceptualized by me (Abhiram Kaushik) but implemented using the Gemini LLM. Permalink to the chat [here](https://share.gemini.google/WEDBVBqp6mVB)

The documentation below is also generated using Gemini. Use with caution (although I will get around to checking it later).

---

## `orchestrator.py`

`orchestrator.py` parses a JSON configuration file, performs pre-flight schema and file existence checks, resolves parameter spaces, and generates self-contained Bash submission scripts (`.sh`).

For multi-dimensional parameter scans, it generates a single **SLURM Job Array** submission script (`submit_array.sh`) using embedded Bash parameter arrays and `$SLURM_ARRAY_TASK_ID` dynamic mapping.

### Command-Line Usage

```bash
# Run with default configuration file (config.json)
python3 orchestrator.py

# Run with a custom configuration file
python3 orchestrator.py path/to/my_config.json

# Display help information
python3 orchestrator.py --help
```

### Script Execution Flow
1. **Load Config:** Reads the target JSON configuration file.
2. **Pre-Flight Validation:**
   * Validates required sections (`execution`, `slurm`, `inner_loop`).
   * Validates data types for flags, environment variables, modules, and arguments.
   * Asserts existence of all referenced inner-loop input files (`file_path` or `inner_files` mappings) on disk prior to script generation. *This functionality is temporarily disabled*
3. **Determine Output Mode:**
   * **Single Run Mode** (`loopQ: false`): Writes `submit_single.sh`.
   * **Inner-Loop Only Mode** (`loopQ: true`, no `outer_loops`): Writes `submit_inner.sh`.
   * **Job Array Mode** (`loopQ: true`, with `outer_loops`): Calculates Cartesian product combinations and writes a unified `submit_array.sh`.
4. **Script Generation:** Embeds timing metrics (`date +%s%N`), environment variable exports, module loads, and `sed`-based output prefixing (`[A<task>_L<line>_out]` / `[A<task>_L<line>_err]`).

---

## Configuration Schema (`config.json`)

The input JSON file controls execution behavior, resource allocation, schema constraints, and parameter space expansion.

### Schema Fields

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `loopQ` | Boolean | Optional | Set to `false` for single execution pass. Defaults to `true` for parameter scans. |
| `execution` | Object | **Yes** | Language binary, executable, flags, modules, and environment variables. |
| `slurm` | Object | **Yes** | Key-value pairs translated into `#SBATCH --<key>=<value>` directives. |
| `slurm.max_concurrent_tasks` | Integer | Optional | Throttles maximum active array tasks on the cluster (e.g., `#SBATCH --array=0-7%2`). |
| `inner_loop` | Object | Conditional | Required if `loopQ: true`. Specifies kinematic data source and CLI argument mapping. |
| `outer_loops` | Array | Optional | Used in `loopQ: true` mode for multidimensional sweeps converted to Job Arrays. |
| `args` | Object | Conditional | Required if `loopQ: false`. Command-line key-value pairs for a single run. |

---

## Configuration Modes & Examples

### Case 1: Full Parameter Scan (SLURM Job Array Mode)

Used when scanning across multiple outer-loop parameters while reading space-separated kinematic data points from dynamic or static inner-loop files.

```json
{
  "loopQ": true,
  "execution": {
    "language": "python3",
    "executable": "demo.py",
    "flags": ["-u"],
    "modules": ["gcc/11.2.0", "python/3.10"],
    "env_vars": {
      "JULIA_NUM_THREADS": "$SLURM_CPUS_PER_TASK",
      "OMP_NUM_THREADS": "4"
    }
  },
  "slurm": {
    "job-name": "dis_full_scan",
    "partition": "standard",
    "nodes": 1,
    "cpus-per-task": 4,
    "time": "02:00:00",
    "output": "slurm_%A_%a.log",
    "max_concurrent_tasks": 2
  },
  "inner_loop": {
    "file_path": null,
    "delimiter": " ",
    "arg_names": ["x", "Q2"]
  },
  "outer_loops": [
    {
      "arg_name": "dipole_fit",
      "values": ["gbw", "mv"],
      "inner_files": {
        "gbw": "data/kinematics_gbw.txt",
        "mv": "data/kinematics_mv.txt"
      }
    },
    {
      "arg_name": "scale",
      "values": [0.5, 1.0]
    }
  ]
}
```

* **Output File:** `submit_array.sh`
* **Generated Header:** `#SBATCH --array=0-3%2`
* **Behavior:** Maps outer combinations into Bash parameter arrays (`OUTER_ARGS`, `TARGET_FILES`). The array task index (`$SLURM_ARRAY_TASK_ID`) dynamically selects parameter arguments and target data files at runtime.

---

### Case 2: Inner Loop Only (Single Script Scan)

Used when running a 1D sweep over a single file without outer-loop parameters.

```json
{
  "loopQ": true,
  "execution": {
    "language": "python3",
    "executable": "demo.py",
    "flags": ["-u"],
    "modules": ["python/3.10"],
    "env_vars": {}
  },
  "slurm": {
    "job-name": "dis_inner_scan",
    "partition": "standard",
    "nodes": 1,
    "time": "01:00:00",
    "output": "slurm_%j.log"
  },
  "inner_loop": {
    "file_path": "data/kinematics_single.txt",
    "delimiter": " ",
    "arg_names": ["x", "Q2"]
  }
}
```

* **Output File:** `submit_inner.sh`
* **Log Prefix Format:** `[L1_out]`, `[L2_out]`, etc.

---

### Case 3: Single Run (`loopQ: false`)

Used to run the executable once with a static set of CLI arguments without external data files.

```json
{
  "loopQ": false,
  "execution": {
    "language": "python3",
    "executable": "demo.py",
    "flags": ["-u"],
    "modules": ["python/3.10"],
    "env_vars": {
      "OMP_NUM_THREADS": "1"
    }
  },
  "slurm": {
    "job-name": "dis_single",
    "partition": "short",
    "nodes": 1,
    "time": "00:15:00",
    "output": "slurm_%j.log"
  },
  "args": {
    "dipole_fit": "gbw",
    "scale": 1.0,
    "x": 0.0001,
    "Q2": 2.5
  }
}
```

* **Output File:** `submit_single.sh`
* **Log Prefix Format:** `[SINGLE_out]` and `[SINGLE_err]`.

---

## Log Output & Stream Tagging

Generated scripts print an explicit run legend at execution start, followed by real-time stream tagging and nano-second precision job timing:

```text
======================= SLURM ARRAY RUN LEGEND =======================
Array Task ID: 0
Outer Loop Combo: dipole_fit=gbw, scale=0.5
Outer Flags: --dipole_fit gbw --scale 0.5
Inner Loop Source File: data/kinematics_gbw.txt
Inner Args: x, Q2
======================================================================
[A0_L1_out] Job started at: Fri Aug  7 14:00:00 EEST 2026
[A0_L1_out] Running point x=0.0001 Q2=1.0...
[A0_L1_out] Job finished at: Fri Aug  7 14:00:02 EEST 2026
[A0_L1_out] Job duration: 2.104 seconds
[A0_L2_out] Job started at: Fri Aug  7 14:00:02 EEST 2026
[A0_L2_err] Warning: Convergence threshold near limit.
[A0_L2_out] Job finished at: Fri Aug  7 14:00:04 EEST 2026
[A0_L2_out] Job duration: 1.980 seconds
```