# SLURM Execution Orchestrator

An automated framework for generating and submitting SLURM batch scripts for scientific compute codes, supporting single runs, 1D parameter loops, and multi-dimensional outer/inner parameter scans.

---

## `orchestrator.py`

`orchestrator.py` parses a JSON configuration file, resolves parameter spaces, and generates self-contained Bash submission scripts (`.sh`) configured with SLURM `#SBATCH` directives, environment module loads, environment variable exports, execution timing, and tagged `stdout`/`stderr` line prefixing.

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
2. **Determine Mode:** Evaluates `loopQ` setting to select **Single Run**, **Inner-Loop Only**, or **Full Scan** mode.
3. **Environment Setup:** Generates `module load` statements and `export` statements for environment variables.
4. **Script Generation:** Constructs the bash submission script(s) with embedded execution timing and `sed`-based output stream prefixing (`[..._out]` and `[..._err]`).
5. **Output:** Writes submission script files (`submit_single.sh`, `submit_inner.sh`, or `submit_O<idx>.sh`).

---

## Configuration Schema (`config.json`)

The input JSON file controls execution behavior, resource allocation, and parameter space expansion.

### Schema Fields

| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `loopQ` | Boolean | Optional | Set to `false` for a single execution pass. Set to `true` (or omit) for looped sweeps. |
| `execution` | Object | **Yes** | Defines language binary, executable script, execution flags, modules, and environment variables. |
| `slurm` | Object | **Yes** | Key-value pairs translated directly into `#SBATCH --<key>=<value>` directives. |
| `inner_loop` | Object | Conditional | Required if `loopQ: true`. Specifies kinematic data source and CLI argument mapping. |
| `outer_loops` | Array | Optional | Used in `loopQ: true` mode for multidimensional sweeps over parameter lists. |
| `args` | Object | Conditional | Required if `loopQ: false`. Command-line arguments for a single run. |

---

## Configuration Modes & Examples

### Case 1: Full Parameter Scan (Outer + Inner Loops)

Used when scanning across multiple outer-loop parameters (e.g., dipole fits, scales) while looping over space-separated kinematic points from a file in the inner loop.

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
    "output": "slurm_%j.log"
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

* **Behavior:** Generates `submit_O0.sh`, `submit_O1.sh`, etc., for every Cartesian product combination of `outer_loops`.
* **File Resolution:** Because `inner_loop.file_path` is `null`, it retrieves the file from `inner_files` corresponding to the current `dipole_fit` value.

---

### Case 2: Inner Loop Only (No Outer Loops)

Used when executing a parameter sweep over a single file without any outer-loop parameters.

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

* **Behavior:** Generates a single `submit_inner.sh` script.
* **Log Prefix Format:** Output tags drop the `O<idx>` prefix and format directly as `[L1_out]`, `[L2_out]`, etc.

---

### Case 3: Single Run (`loopQ: false`)

Used to run the executable once with a static set of parameters without reading external data files.

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

* **Behavior:** Generates `submit_single.sh`.
* **Log Prefix Format:** Log lines are tagged with `[SINGLE_out]` and `[SINGLE_err]`.

---

## Log Output & Legend Tracking

Every generated script prints a **Run Legend** at the top of the SLURM output log, followed by real-time prefixed execution streams and timing metrics:

```text
======================= SLURM RUN LEGEND =======================
Outer Loop Combo: dipole_fit=gbw, scale=0.5
Outer Flags: --dipole_fit gbw --scale 0.5
Inner Loop Source File: data/kinematics_gbw.txt
Inner Args: x, Q2
================================================================
[O0_L1_out] Job started at: Fri Aug  7 14:00:00 EEST 2026
[O0_L1_out] Running point x=0.0001 Q2=1.0...
[O0_L1_out] Job finished at: Fri Aug  7 14:00:02 EEST 2026
[O0_L1_out] Job duration: 2.104 seconds
[O0_L2_out] Job started at: Fri Aug  7 14:00:02 EEST 2026
[O0_L2_err] Warning: Approach limit reached.
[O0_L2_out] Job finished at: Fri Aug  7 14:00:04 EEST 2026
[O0_L2_out] Job duration: 1.980 seconds
```
