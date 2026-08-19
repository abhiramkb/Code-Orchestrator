import json
from datetime import datetime
import shutil
import itertools
import argparse
import sys
from pathlib import Path
import subprocess
import re
import os
import sqlite3
from typing import Any, Dict, Optional

# Required for Pydantic based validation of input config file
from config_schema import validate_config, AppConfig, determine_loop_q, ExperimentConfig, SlurmConfig, ExecutionConfig

# Required for constructing mode-specific portion of SLURM scripts
from mode_builders import build_single_mode, build_inner_loop_mode, build_job_array_mode

def get_dict_from_config_file(config_path) -> dict:
    config_file = Path(config_path)
    if not config_file.is_file():
        print(f"[ERROR] Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(config_file) as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in '{config_path}': {e}", file=sys.stderr)
            sys.exit(1)
    return cfg



def format_cli_args(args_dict: Dict[str, Any]) -> str:
    """Formats key-value argument pairs into standard CLI flag strings."""
    args_list = []
    for k, v in args_dict.items():
        if isinstance(v, str):
            args_list.append(f'--{k} \\"{v}\\"')
        else:
            args_list.append(f'--{k} {v}')
    return " ".join(args_list)

def build_experiment_strings(experiment: Optional[ExperimentConfig]) -> tuple[str, str, str]:
    """
    Builds the environment block and argument block for experiment tracking.
    Returns (exp_name, exp_env_block, exp_args_block).
    """
    if experiment is None:
        exp_env_block = """
export CHECKPOINT_DIR="./checkpoints"
mkdir -p "$CHECKPOINT_DIR"
"""
        exp_args_block = '\n    exp_args=""'
        return "noexp", exp_env_block, exp_args_block

    db_path = experiment.result_database_path
    exp_name = experiment.experiment_name
    slrm_output_dir_name = experiment.slrm_output_dir

    exp_env_block = f"""
# =========================================================================
# EXPERIMENT TRACKING
# =========================================================================
export RESULT_DATABASE_PATH="{db_path}"
export EXPERIMENT_NAME="{exp_name}"
export SLRM_OUTPUT_DIR="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/{slrm_output_dir_name}"

# Safe directory creation for the specific job
export JOB_ID=${{SLURM_JOB_ID:-$$}}
export SAVE_DIR="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/$JOB_ID"
export CHECKPOINT_DIR="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/checkpoints"

mkdir -p "$SAVE_DIR"
mkdir -p "$CHECKPOINT_DIR"
"""

    tracking_args = experiment.tracking_args
    arg_components = []
    if tracking_args is not None:
        for key, val_template in tracking_args.items():
            formatted_val = (
                val_template
                .replace("{result_database_path}", "${RESULT_DATABASE_PATH}")
                .replace("{experiment_name}", "${EXPERIMENT_NAME}")
                .replace("{slrm_output_dir}", "${SLRM_OUTPUT_DIR}")
                .replace("{save_dir}", "${SAVE_DIR}")
                .replace("{__indicator}", "${indicator}")
            )
            arg_components.append(f'--{key} \\"{formatted_val}\\"')
    tracking_args_str = " ".join(arg_components)

    exp_args_block = f"""
    exp_args=""
    if [ -n "${{SAVE_DIR:-}}" ]; then
        exp_args="{tracking_args_str}"
    fi"""
    return exp_name, exp_env_block, exp_args_block

def setup_experiment_directories(
    experiment: Optional[ExperimentConfig],
    slurm: SlurmConfig,
    config_file: Path,
    dryrun_q: bool
) -> None:
    """
    Creates SLURM output directory and backs up the config file if dryrun is False.
    Updates slurm.output directly on the SlurmConfig model.
    """
    if experiment is None:
        return

    db_path = experiment.result_database_path
    exp_name = experiment.experiment_name
    slrm_output_dir_name = experiment.slrm_output_dir

    full_slrm_output_dir = f"{db_path}/{exp_name}/{slrm_output_dir_name}"
    # Mutates the Pydantic model directly
    slurm.output = f"{full_slrm_output_dir}/{exp_name}_%j.out"

    try:
        Path(full_slrm_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Ensured SLURM output directory exists: {full_slrm_output_dir}")
    except Exception as e:
        print(f"[WARNING] Could not create SLRM_OUTPUT_DIR '{full_slrm_output_dir}': {e}", file=sys.stderr)

    if not dryrun_q:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(db_path) / exp_name / datetime_str
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_file, backup_dir / config_file.name)
            print(f"[INFO] Copied config file to: {backup_dir / config_file.name}")
        except Exception as e:
            print(f"[WARNING] Could not copy config file to '{backup_dir}': {e}", file=sys.stderr)

def build_common_header(slurm: SlurmConfig, exec_cfg: ExecutionConfig, exp_env_block: str) -> str:
    """Generates the SBATCH headers, environment modules, export variables, and print helper function."""
    # Dump SlurmConfig using aliases (e.g. 'job-name') and excluding None values
    slurm_dict = slurm.model_dump(by_alias=True, exclude_none=True)
    
    # Exclude internal configuration fields not meant for SBATCH
    slurm_dict.pop("max_concurrent_tasks", None)

    slurm_header = "".join(f'#SBATCH --{k}={v}\n' for k, v in slurm_dict.items())

    modules = exec_cfg.modules
    module_load_block = (
        "\n# Load Required Environment Modules\n" + "\n".join(f"module load {m}" for m in modules)
        if modules else "# No environment modules specified"
    )

    env_vars = exec_cfg.env_vars
    env_var_block = (
        "\n# Set Environment Variables\n" + "\n".join(f'export {k}={v}' for k, v in env_vars.items())
        if env_vars else "# No environment variables specified"
    )

    print_args_def = """
# Function to parse and display command-line arguments to SLURM output
print_args() {
    local ind="$1"
    shift
    local args_str="$*"
    if [ -z "$args_str" ]; then
        return
    fi
    eval "set -- $args_str"
    while [ $# -gt 0 ]; do
        case "$1" in
            --*)
                local param_name="${1#--}"
                if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
                    echo "[${ind}_args] ${param_name}: $2"
                    shift 2
                else
                    echo "[${ind}_args] ${param_name}: "
                    shift 1
                fi
                ;;
            *)
                shift 1
                ;;
        esac
    done
}
"""
    return f"""#!/bin/bash
{slurm_header}
{exp_env_block}
{module_load_block}
{env_var_block}
{print_args_def}"""


def write_script(script_content: str, mode_prefix: str, exp_name: str, timestamp: str) -> Path:
    """
    Writes the script content to a file and prints the location.
    mode_prefix is one of 'single', 'inner', 'array'.
    """
    script_file = Path(f"submit_{mode_prefix}_{exp_name}_{timestamp}.sh")
    script_file.write_text(script_content)
    print(f"Generated {mode_prefix}-run SLURM script: {script_file}")
    return script_file



def generate_slurm_script(config_path, dryrunQ):
    cfg = get_dict_from_config_file(config_path)
    config_file = Path(config_path)

    # 1. Validate & convert raw dict to typed AppConfig object
    config: AppConfig = validate_config(cfg, config_path, dryrunQ)
    print(f"[SUCCESS] Config validation passed for '{config_path}'.")

    # 2. Extract execution metadata cleanly via dot-notation
    loop_q = determine_loop_q(cfg)
    exec_cfg = config.execution
    slurm_cfg = config.slurm
    experiment_cfg = config.experiment
    max_concurrent = slurm_cfg.max_concurrent_tasks  # Defined on SlurmConfig model

    # 3. Filesystem setup (directories + config backup)
    setup_experiment_directories(experiment_cfg, slurm_cfg, config_file, dryrunQ)

    # 4. Build experiment strings & execution context
    exp_name, exp_env_block, exp_args_block = build_experiment_strings(experiment_cfg)
    common_header = build_common_header(slurm_cfg, exec_cfg, exp_env_block)

    exec_path = Path(exec_cfg.executable).resolve()
    exec_dir = exec_path.parent
    fixed_args_str = format_cli_args(config.args)
    flags_str = " ".join(exec_cfg.flags)
    exec_sig_components = [exec_cfg.language, flags_str, str(exec_path), fixed_args_str]
    exec_sig_str = " ".join(p for p in exec_sig_components if p)

    ctx = {
        "exec_cfg": exec_cfg,
        "exec_path": exec_path,
        "exec_dir": exec_dir,
        "flags_str": flags_str,
        "fixed_args_str": fixed_args_str,
        "exec_sig_str": exec_sig_str,
        "exp_args_block": exp_args_block,
        "common_header": common_header,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 5. Clean Mode Routing via computed config properties
    mode_index = config.mode_index

    if mode_index == 0:
        # Mode 0: Single Run
        script_content = build_single_mode(ctx, config)
        return write_script(script_content, "single", exp_name, timestamp)

    elif mode_index == 1:
        # Mode 1: Inner Loop Only
        script_content = build_inner_loop_mode(ctx, config)
        return write_script(script_content, "inner", exp_name, timestamp)

    else:
        # Mode 2: Array Mode (Outer + Inner Loops)
        script_content, total_tasks = build_job_array_mode(ctx, config, max_concurrent)
        script_file = write_script(script_content, "array", exp_name, timestamp)
        print(f"({total_tasks} tasks)")
        return script_file

def extract_config_flags(cfg: dict) -> tuple[set[str], set[str]]:
    """Collects all argument keys across all configuration sections and converts them to CLI flags.
    CAUTION: Code assumes single letter args also use the "--<arg>" convention.
    """
    raw_keys = set()

    # 1. Static arguments
    if isinstance(cfg.get("args"), dict):
        raw_keys.update(cfg["args"].keys())

    # 2. Inner loop argument names
    inner_cfg = cfg.get("inner_loop", {})
    if isinstance(inner_cfg, dict) and isinstance(
        inner_cfg.get("arg_names"), list
    ):
        raw_keys.update(inner_cfg["arg_names"])

    # 3. Outer loop argument names
    outer_cfg = cfg.get("outer_loops", [])
    if isinstance(outer_cfg, list):
        for o_loop in outer_cfg:
            if isinstance(o_loop, dict):
                if isinstance(o_loop.get("arg_names"), list):
                    raw_keys.update(o_loop["arg_names"])
                elif isinstance(o_loop.get("arg_name"), str):
                    raw_keys.add(o_loop["arg_name"])

    # 4. Experiment tracking arguments
    exp_cfg = cfg.get("experiment", {})
    if isinstance(exp_cfg, dict) and isinstance(
        exp_cfg.get("tracking_args"), dict
    ):
        raw_keys.update(exp_cfg["tracking_args"].keys())

    # Convert keys into CLI flag formats (e.g., "lr" -> "--lr", "v" -> "-v")
    formatted_flags = set()
    for key in raw_keys:
        key_str = str(key).strip()
        if not key_str.startswith("-"):
            #flag = f"-{key_str}" if len(key_str) == 1 else f"--{key_str}"
            flag = f"--{key_str}"
        else:
            flag = key_str
        formatted_flags.add(flag)

    return formatted_flags, raw_keys

def validate_script_args(config_path) -> tuple[bool, list[str]]:
    """Validates if arguments defined in the config dictionary are supported by the target executable via --help.

    Returns:
        tuple[bool, list[str]]: (is_valid, list_of_unsupported_flags)
    """
    cfg = get_dict_from_config_file(config_path)
    
    exec_cfg = cfg.get("execution", {})
    interpreter = exec_cfg.get("language", "")
    executable_path = exec_cfg.get("executable", "")
    modules = exec_cfg.get("modules", [])

    if not interpreter or not executable_path:
        raise ValueError(
            "Missing required 'execution.language' or 'execution.executable' in config."
        )

    # Collect flags planned across all modes
    flags_to_check, _ = extract_config_flags(cfg)
    if not flags_to_check:
        return True, []

    # Build bash command to load modules and invoke help flag
    module_cmds = [f"module load {m}" for m in modules]
    exec_cmd = f"{interpreter} {executable_path} --help"
    full_cmd = " && ".join(module_cmds + [exec_cmd])

    try:
        result = subprocess.run(
            ["bash", "-c", full_cmd],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        help_output = result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr_msg = e.stderr if hasattr(e, "stderr") else str(e)
        raise RuntimeError(
            f"Failed to execute '--help' check on '{executable_path}':\n{stderr_msg}"
        ) from e

    # Extract all supported option flags from --help text
    raw_found_flags = set(re.findall(r"(?<!\w)(-[a-zA-Z0-9_|-]+)", help_output))

    supported_flags = set()
    for flag_group in raw_found_flags:
        for flag in re.split(r"[,|/]", flag_group):
            clean_flag = flag.strip()
            supported_flags.add(clean_flag)

    # Check each planned flag (handles underscore vs hyphen conversion e.g. --batch_size vs --batch-size)
    unsupported = []
    for flag in flags_to_check:
        alt_hyphen = flag.replace("_", "-")
        alt_underscore = flag.replace("-", "_")

        if not ({flag, alt_hyphen, alt_underscore} & supported_flags):
            unsupported.append(flag)

    return len(unsupported) == 0, unsupported

def submit_slurm_script(script_path: Path, config_path, checkargsQ):
    """Submits the generated script to SLURM."""

    if checkargsQ:
        print(f"\n[INFO] Validating arguments passed to executable...")
        print(f"[INFO] This may take some time. Use --noargcheck to disable argument checking.")
        print(f"[INFO] CAUTION: Validation function expects all args in \"--<args>\" format!")
        print(f"[INFO] This includes single letter args!")
        validation_result, failed_args = validate_script_args(config_path)
        if not validation_result:
            print(f"\n[ERROR] Argument validation failed!")
            print(f"[ERROR] Offending args: ", " ".join(failed_args))
            sys.exit(1)
        else:
            print(f"\n[SUCCESS] Argument validation succeeded!")
    
    print(f"\n[INFO] Submitting {script_path} to SLURM...")
    try:
        # Calls sbatch. The embedded checkpoint logic handles skips securely.
        result = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] SLURM Submission failed:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] 'sbatch' command not found. Are you on a SLURM cluster node?", file=sys.stderr)
        sys.exit(1)

# Collection function from https://share.gemini.google/ldX5zTud8zOv, https://share.gemini.google/IEM7GOQgmFCx
# Note that this function is specific to the NLODiffraction project. A more general collection function needs
# to be developed.
def collect_slurm_results_to_db(config_path: str) -> None:
    """Collects SLURM job execution results and metadata into a SQLite database.

    Args:
        config_path (str): Path to the orchestrator JSON configuration file.
    """
    # 1. Load configuration file
    config = get_dict_from_config_file(config_path)

    exp_cfg = config["experiment"]
    result_db_path = exp_cfg["result_database_path"]
    exp_name = exp_cfg["experiment_name"]
    slurm_output_dir = exp_cfg["slrm_output_dir"]

    # Parent directory for results and SLURM logs path
    exp_dir = os.path.join(result_db_path, exp_name)
    slurm_dir = os.path.join(exp_dir, slurm_output_dir)
    db_file_path = os.path.join(exp_dir, "results.db")

    if not os.path.exists(slurm_dir):
        raise FileNotFoundError(
            f"SLURM output directory not found: {slurm_dir}"
        )

    records = []

    # Helper function to recursively extract flat key-value pairs
    def flatten_json(data):
        flat = {}
        for k, v in data.items():
            if isinstance(v, dict):
                flat.update(flatten_json(v))
            else:
                flat[k] = v
        return flat

    # 2. Iterate through files in the SLURM output directory
    for filename in os.listdir(slurm_dir):
        file_path = os.path.join(slurm_dir, filename)
        if not os.path.isfile(file_path):
            continue
        else:
            print("Processing file: ",filename)

        # Extract job_id from filename pattern: <experiment_name>_<job_id>
        job_id_match = re.search(rf"{re.escape(exp_name)}_(\d+)", filename)
        if not job_id_match:
            continue
        base_job_id = job_id_match.group(1)
        print("Extracted job_id: ", base_job_id)

        # 3. Parse SLURM log file for ALL (indicator, duration) instances
        sub_jobs = []
        with open(file_path, "r") as log_file:
            for line in log_file:
                dur_match = re.search(
                    r"\[(.*?)_out\]\s*Job duration:\s*([\d\.]+)\s*seconds", line
                )
                if dur_match:
                    indicator = dur_match.group(1)
                    duration = float(dur_match.group(2))
                    sub_jobs.append((indicator, duration))
                # In case some jobs are skipped over due to an existing checkpoint but somehow they still have subfolders matching the current job id (not sure how that happened but here we are), I want to check the indicators that have been skipped, add them to the sub_jobs list with the duration as NULL.
                checkpoint_match = re.search(
                    r"\[CHECKPOINT\]\sSkipping\s(.*?) - already completed\.", line
                )
                if checkpoint_match:
                    indicator = checkpoint_match.group(1)
                    duration = None
                    print(
                        f"[Warning] Recording \"Null\" duration for skipped job"
                    )
                    sub_jobs.append((indicator, duration))


        if not sub_jobs:
            print(
                f"[Warning] No completed job durations found in log: {filename}"
            )
            continue
        # 4. Process each sub-job instance found in the log file
        for indicator, duration in sub_jobs:
            target_dir_name = f"{base_job_id}_{indicator}"
            target_dir = os.path.join(exp_dir, target_dir_name)

            # Direct fallback check if directory name differs
            if not os.path.exists(target_dir):
                alt_target_dir = os.path.join(exp_dir, indicator)
                if os.path.exists(alt_target_dir):
                    target_dir = alt_target_dir
                else:
                    print(
                        f"[Warning] Target result directory not found: {target_dir}"
                    )
                    continue

            # 5. Locate and flatten the JSON file starting with "result"
            json_file_path = None
            for fname in os.listdir(target_dir):
                if fname.startswith("result") and fname.endswith(".json"):
                    json_file_path = os.path.join(target_dir, fname)
                    break

            if not json_file_path or not os.path.exists(json_file_path):
                print(
                    f"[Warning] No result JSON file found in directory: {target_dir}"
                )
                continue

            with open(json_file_path, "r") as jf:
                json_raw = json.load(jf)

            flat_data = flatten_json(json_raw)

            # Store job identifier as <job_id>_<indicator> (e.g. 590721_L1)
            job_identifier = f"{base_job_id}_{indicator}"

            records.append(
                {
                    "job_id": job_identifier,
                    "json_data": flat_data,
                    "job_duration": duration,
                }
            )

    if not records:
        print("No valid job execution records found to write to database.")
        return

    # 6. Prepare unique dynamic columns and infer their SQLite types
    key_types = {}
    for rec in records:
        for key, val in rec["json_data"].items():
            if key not in key_types or key_types[key] == "TEXT":
                if isinstance(val, bool):
                    # SQLite doesn't have a native BOOL type; stored as INTEGER (0/1)
                    key_types[key] = "INTEGER"
                elif isinstance(val, int):
                    key_types[key] = "INTEGER"
                elif isinstance(val, float):
                    key_types[key] = "REAL"
                else:
                    key_types[key] = "TEXT"

    all_json_keys = list(key_types.keys())
    column_names = ["job_id"] + all_json_keys + ["job_duration"]

    # 7. Create/Update SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    col_definitions = ['"job_id" TEXT PRIMARY KEY']
    for k in all_json_keys:
        col_type = key_types.get(k, "TEXT")
        col_definitions.append(f'"{k}" {col_type}')
    col_definitions.append('"job_duration" REAL')

    create_table_sql = (
        f"CREATE TABLE IF NOT EXISTS runs ({', '.join(col_definitions)});"
    )
    cursor.execute(create_table_sql)

    quoted_cols = [f'"{c}"' for c in column_names]
    placeholders = ["?" for _ in column_names]

    insert_sql = f"""
        INSERT OR REPLACE INTO runs ({', '.join(quoted_cols)})
        VALUES ({', '.join(placeholders)})
    """

    rows_to_insert = []
    for rec in records:
        row = [rec["job_id"]]
        for key in all_json_keys:
            val = rec["json_data"].get(key, None)
            if isinstance(val, (dict, list)):
                val = json.dumps(val)
            elif isinstance(val, bool):
                val = int(val)
            row.append(val)
        row.append(rec["job_duration"])
        rows_to_insert.append(row)

    cursor.executemany(insert_sql, rows_to_insert)

    conn.commit()
    conn.close()
    print(
        f"Successfully written {len(rows_to_insert)} run(s) to database: {db_file_path}"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and submit SLURM scripts from a JSON config.")
    parser.add_argument(
        "config",
        nargs="?",
        default="config.json",
        help="Path to the JSON configuration file (default: config.json)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Automatically submit the generated script to SLURM."
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Dry run."
    )
    parser.add_argument(
        "--noargcheck",
        action="store_true",
        help="Disable checking of whether args passed to the executable are supported."
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collects results from finished SLURM jobs."
    )
    args = parser.parse_args()


    #cfg = get_dict_from_config_file(args.config)
    #_,result = extract_config_flags(cfg)
    #print(result)
    #exit()

    if args.collect:
        collect_slurm_results_to_db(args.config)
        exit()

    checkargsQ = not args.noargcheck
    
    generated_script = generate_slurm_script(args.config, args.dryrun)

    if args.dryrun:
        print(f"\n[INFO] Dry-run: not submitting to SLURM.")
    elif args.submit:
        submit_slurm_script(generated_script, args.config, checkargsQ)
    else:
        print(f"\n[TIP] Run 'sbatch {generated_script}' to manually submit, or pass --submit next time.")