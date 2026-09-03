import json
import yaml
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
from typing import Any, Dict, Optional, List, Set, Tuple

# Required for Pydantic based validation of input config file
from config_schema import (
    validate_config, AppConfig, ExperimentConfig, SlurmConfig, ExecutionConfig,
    TabularInnerLoop, TabularOuterLoop, get_slurm_cpus_per_task, parse_slurm_mem_mb,
    parse_slurm_time_seconds,
)

# Required for constructing mode-specific portion of SLURM scripts
from mode_builders import build_single_mode, build_inner_loop_mode, build_job_array_mode

# Utilities
from utils import format_cli_args

def get_dict_from_config_file(config_path) -> dict:
  config_file = Path(config_path)
  if not config_file.is_file():
    print(
        f"[ERROR] Configuration file '{config_path}' not found.",
        file=sys.stderr,
    )
    sys.exit(1)

  suffix = config_file.suffix.lower()

  with open(config_file, "r", encoding="utf-8") as f:
    try:
      if suffix == ".json":
        cfg = json.load(f)
      elif suffix in (".yaml", ".yml"):
        cfg = yaml.safe_load(f)
      else:
        # Fallback for unrecognized extensions: YAML parses valid JSON natively
        cfg = yaml.safe_load(f)
    except (json.JSONDecodeError, yaml.YAMLError) as e:
      print(
          f"[ERROR] Failed to parse configuration file '{config_path}': {e}",
          file=sys.stderr,
      )
      sys.exit(1)

  if not isinstance(cfg, dict):
    print(
        f"[ERROR] Configuration in '{config_path}' must be a valid YAML or JSON file interpretable as a key-value mapping"
        " (dictionary).",
        file=sys.stderr,
    )
    sys.exit(1)

  return cfg


def build_experiment_strings(experiment: Optional[ExperimentConfig], mode_index) -> Tuple[str, str, str]:
    """
    Builds the environment block and argument block for experiment tracking.
    Returns (exp_name, exp_env_block, exp_args_block).
    """
    if experiment is None:
        exp_env_block = """
# Absolute so it still resolves after the script cd's to the executable directory
export CHECKPOINT_DIR="$(pwd)/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
"""
        exp_args_block = '\n    exp_args=""'
        return "noexp", exp_env_block, exp_args_block

    db_path = experiment.result_database_path
    exp_name = experiment.experiment_name
    slrm_output_dir_name = experiment.slrm_output_dir

    job_id_var = "SLURM_ARRAY_JOB_ID" if mode_index == 2 else "SLURM_JOB_ID"

    exp_env_block = f"""
# =========================================================================
# EXPERIMENT TRACKING
# =========================================================================
export RESULT_DATABASE_PATH="{db_path}"
export EXPERIMENT_NAME="{exp_name}"
export SLRM_OUTPUT_DIR="$RESULT_DATABASE_PATH/$EXPERIMENT_NAME/{slrm_output_dir_name}"

# Safe directory creation for the specific job
export JOB_ID=${{{job_id_var}:-$$}}
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
    mode_index: int,
    dryrun_q: bool,
    timestamp: str,
) -> Optional[Path]:
    """
    Creates the SLURM output directory and updates slurm.output on the model.

    Returns the path the run's backup directory *should* take, without creating
    it. Creation is deferred to write_script, so a run that turns out to have
    nothing to submit - or whose script fails its syntax check - leaves no
    stray timestamped directory behind.
    """
    if experiment is None:
        return

    db_path = experiment.result_database_path
    exp_name = experiment.experiment_name
    slrm_output_dir_name = experiment.slrm_output_dir

    full_slrm_output_dir = f"{db_path}/{exp_name}/{slrm_output_dir_name}"
    # Mutates the Pydantic model directly
    slurm.output = f"{full_slrm_output_dir}/{exp_name}_%j.out"
    if mode_index == 2:
        # For array mode, we want to include the indicator in the output filename
        slurm.output = f"{full_slrm_output_dir}/{exp_name}_%A_%a.out"
    try:
        Path(full_slrm_output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Ensured SLURM output directory exists: {full_slrm_output_dir}")
    except Exception as e:
        print(f"[WARNING] Could not create SLRM_OUTPUT_DIR '{full_slrm_output_dir}': {e}", file=sys.stderr)

    if dryrun_q:
        return None
    return Path(db_path) / exp_name / timestamp
  
def _get_generator_git_commit() -> str:
    """Retrieves the git commit hash of the file's repository."""
    try:
        file_dir = Path(__file__).resolve().parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=file_dir,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "UNKNOWN (not a git repo or git not installed)"

def build_common_header(
    slurm: SlurmConfig,
    exec_cfg: ExecutionConfig,
    exp_env_block: str,
    timestamp: str = "",
    backup_dir: Optional[Path] = None,
) -> str:
    """Generates the SBATCH headers, environment modules, export variables, and print helper function."""
    generator_commit = _get_generator_git_commit()

    # Dump SlurmConfig using aliases (e.g. 'job-name') and excluding None values
    slurm_dict = slurm.model_dump(by_alias=True, exclude_none=True)
    
    # Exclude internal configuration fields not meant for SBATCH
    slurm_dict.pop("max_concurrent_tasks", None)
    slurm_dict.pop("num_array_jobs", None)

    slurm_header = "".join(f'#SBATCH --{k}={v}\n' for k, v in slurm_dict.items())

    modules = exec_cfg.modules
    module_load_block = (
        "\n# Load Required Environment Modules\n" + "\n".join(f"module load {m}" for m in modules)
        if modules else "# No environment modules specified"
    )

    # Exported before env_vars so a config can set e.g. JULIA_NUM_THREADS to
    # "$MULTITHREADING_LEVEL" and get the per-run thread count.
    level_block = (
        f"\n# Cores used by one invocation of the executable\n"
        f"export MULTITHREADING_LEVEL={exec_cfg.multithreading_level}"
        if exec_cfg.multithreading_level is not None else ""
    )

    env_vars = exec_cfg.env_vars
    env_var_block = (
        "\n# Set Environment Variables\n" + "\n".join(f'export {k}={v}' for k, v in env_vars.items())
        if env_vars else "# No environment variables specified"
    )

    preamble = exec_cfg.preamble
    preamble_block = (
        "\n# User-Supplied Preamble\n" + preamble
        if preamble else "# No preamble specified"
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
    # Echoed (not just commented) so the generation timestamp lands in the SLURM
    # output itself, tying a job back to the config/script snapshot that produced
    # it. Carries no '_out]' or 'Job duration:' text, so --collect ignores it.
    provenance_lines = [f'echo "[GENERATION] Script generated at: {timestamp}"'] if timestamp else []
    if backup_dir is not None:
        provenance_lines.append(f'echo "[GENERATION] Config/script snapshot: {backup_dir}"')
    provenance_block = "\n".join(provenance_lines)

    return f"""#!/bin/bash
{slurm_header}

# =========================================================================
# GENERATOR METADATA
# Generated by Orchestrator Commit: {generator_commit}
# Generation timestamp: {timestamp}
# =========================================================================
{provenance_block}

{exp_env_block}
{module_load_block}
{level_block}
{env_var_block}
{preamble_block}
{print_args_def}"""


def warn_about_concurrency(config: AppConfig) -> None:
    """Flags configurations whose thread settings will oversubscribe the node.

    Thread counts stay the user's responsibility (nothing is rewritten), but an
    env var still pinned to $SLURM_CPUS_PER_TASK while the inner loop runs N ways
    concurrently means N x the cores are requested, which is slower than running
    sequentially. Warn rather than block.
    """
    level = config.execution.multithreading_level
    if level is None:
        return

    offenders = [
        k for k, v in config.execution.env_vars.items()
        if "SLURM_CPUS_PER_TASK" in str(v)
    ]
    if offenders:
        print(
            f"[WARNING] {', '.join(offenders)} still reference(s) $SLURM_CPUS_PER_TASK while "
            f"multithreading_level={level} is set.",
            file=sys.stderr,
        )
        print(
            "[WARNING] Each concurrent run would request the whole allocation. Set these to "
            "$MULTITHREADING_LEVEL (exported by the generated script) or a literal value.",
            file=sys.stderr,
        )

    cpus = get_slurm_cpus_per_task(config.slurm)
    if cpus:
        njobs = max(1, cpus // level)
        mem_mb = parse_slurm_mem_mb((config.slurm.model_extra or {}).get("mem"))
        detail = ""
        if mem_mb and njobs > 1:
            detail = f"; --mem is shared, leaving ~{mem_mb // njobs} MB per concurrent run"
        print(f"[INFO] Inner loop will run up to {njobs} concurrent run(s){detail}.")


def get_partition_max_time(partition: str) -> Tuple[Optional[int], Optional[str]]:
    """Looks up a partition's MaxTime via scontrol.

    Returns (max_seconds, error). max_seconds is None when the limit is unknown
    or UNLIMITED. error is set only when the partition genuinely does not exist;
    when scontrol itself is unavailable both are None so the caller skips the
    check (e.g. a --dryrun off the cluster).
    """
    try:
        result = subprocess.run(
            ["scontrol", "show", "partition", partition],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None  # no scontrol here: nothing to check against

    if result.returncode != 0:
        return None, f"partition '{partition}' does not exist on this cluster"

    match = re.search(r"MaxTime=(\S+)", result.stdout)
    if not match:
        return None, None
    return parse_slurm_time_seconds(match.group(1)), None


def validate_partition_time_limit(config: AppConfig) -> None:
    """Rejects a requested time limit the partition would refuse.

    Mirrors the executable argument check: a violation is a hard error rather
    than a warning, since sbatch would reject the submission anyway. Runs at
    generation time because the default workflow generates a script and submits
    it manually later, so a submit-time-only check would miss most usage.
    """
    partition = config.slurm.partition
    requested = parse_slurm_time_seconds(config.slurm.time)

    max_seconds, error = get_partition_max_time(partition)
    if error is not None:
        print(f"\n[ERROR] Partition validation failed: {error}.", file=sys.stderr)
        print("[ERROR] Use --notimecheck to skip this check.", file=sys.stderr)
        sys.exit(1)

    if max_seconds is None or requested is None:
        return

    if requested > max_seconds:
        print(
            f"\n[ERROR] Requested time '{config.slurm.time}' exceeds the limit of "
            f"partition '{partition}'.",
            file=sys.stderr,
        )
        print(
            f"[ERROR] Partition MaxTime is {format_seconds_as_slurm_time(max_seconds)}; "
            f"requested {format_seconds_as_slurm_time(requested)}.",
            file=sys.stderr,
        )
        print("[ERROR] Use --notimecheck to skip this check.", file=sys.stderr)
        sys.exit(1)


def format_seconds_as_slurm_time(total_seconds: int) -> str:
    """Renders seconds as SLURM's D-HH:MM:SS, dropping the day part when zero."""
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    stamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}-{stamp}" if days else stamp


def check_generated_script(script_content: str) -> None:
    """Syntax-checks the generated bash and guards SLURM's script size limit."""
    size = len(script_content.encode("utf-8"))
    if size > 120_000:
        print(
            f"[ERROR] Generated script is {size} bytes; SLURM rejects scripts over 131072. "
            "Reduce the parameter space or split the config.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        result = subprocess.run(
            ["bash", "-n"], input=script_content, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"[WARNING] Could not syntax-check the generated script: {e}", file=sys.stderr)
        return

    if result.returncode != 0:
        print("[ERROR] Generated script failed 'bash -n' syntax check:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)


def write_script(
    script_content: str,
    mode_prefix: str,
    exp_name: str,
    timestamp: str,
    backup_dir: Optional[Path] = None,
    config_file: Optional[Path] = None,
) -> Path:
    """
    Writes the script content to a file and prints the location.
    mode_prefix is one of 'single', 'inner', 'array'.

    Also snapshots the config and the script into the backup directory, which is
    created here rather than earlier so that it only ever exists for a run that
    actually produced a script.
    """
    check_generated_script(script_content)

    script_file = Path(f"submit_{mode_prefix}_{exp_name}_{timestamp}.sh")
    script_file.write_text(script_content)
    print(f"Generated {mode_prefix}-run SLURM script: {script_file}")

    if backup_dir is not None:
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            if config_file is not None:
                shutil.copy2(config_file, backup_dir / config_file.name)
                print(f"[INFO] Copied config file to: {backup_dir / config_file.name}")
            shutil.copy2(script_file, backup_dir / script_file.name)
            print(f"[INFO] Copied SLURM script to: {backup_dir / script_file.name}")
        except Exception as e:
            print(f"[WARNING] Could not populate backup directory '{backup_dir}': {e}", file=sys.stderr)

    return script_file


def discard_backup_dir(backup_dir: Optional[Path]) -> None:
    """Removes a run's backup directory after a submission that produced no job.

    Only ever called for a directory this run created (<db>/<exp>/<timestamp>),
    so there is nothing here that could reach pre-existing results.
    """
    if backup_dir is None or not backup_dir.is_dir():
        return
    try:
        shutil.rmtree(backup_dir)
        print(f"[INFO] Removed backup directory for the failed submission: {backup_dir}")
    except Exception as e:
        print(f"[WARNING] Could not remove backup directory '{backup_dir}': {e}", file=sys.stderr)



def generate_slurm_script(
    config_path, dryrunQ, checktimeQ: bool = True
) -> Tuple[Optional[Path], Optional[Path]]:
    """Generates the submission script. Returns (script_path, backup_dir).

    script_path is None when there is nothing left to submit. backup_dir is
    returned so the caller can discard the snapshot if submission then fails.
    """
    cfg = get_dict_from_config_file(config_path)
    config_file = Path(config_path)

    # One timestamp for the whole run: it names the backup directory, names the
    # script, and is echoed into the SLURM output, so all three always agree.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Validate & convert raw dict to typed AppConfig object
    config: AppConfig = validate_config(cfg, config_path, dryrunQ)
    print(f"[SUCCESS] Config validation passed for '{config_path}'.")
    warn_about_concurrency(config)
    if checktimeQ:
        validate_partition_time_limit(config)

    # 2. Extract execution metadata cleanly via dot-notation
    exec_cfg = config.execution
    slurm_cfg = config.slurm
    experiment_cfg = config.experiment
    max_concurrent = slurm_cfg.max_concurrent_tasks  # Defined on SlurmConfig model

    # 3. Filesystem setup (directories + config backup, function returns backup_dir path if created)
    backup_dir = setup_experiment_directories(
        experiment_cfg, slurm_cfg, config_file, config.mode_index, dryrunQ, timestamp
    )

    # 4. Build experiment strings & execution context
    exp_name, exp_env_block, exp_args_block = build_experiment_strings(experiment_cfg, config.mode_index)
    common_header = build_common_header(
        slurm_cfg, exec_cfg, exp_env_block, timestamp, backup_dir
    )

    exec_path = Path(exec_cfg.executable).resolve()
    exec_dir = exec_path.parent
    fixed_args_str = format_cli_args(config.args)
    flags_str = " ".join(exec_cfg.flags)
    exec_sig_components = [exec_cfg.interpreter, flags_str, str(exec_path), fixed_args_str]
    exec_sig_str = " ".join(p for p in exec_sig_components if p)

    checkpoint_dir = (
        Path(experiment_cfg.checkpoint_dir) if experiment_cfg else Path("checkpoints").resolve()
    )

    ctx = {
        "exec_cfg": exec_cfg,
        "checkpoint_dir": checkpoint_dir,
        "exec_path": exec_path,
        "exec_dir": exec_dir,
        "flags_str": flags_str,
        "fixed_args_str": fixed_args_str,
        "exec_sig_str": exec_sig_str,
        "exp_args_block": exp_args_block,
        "common_header": common_header,
    }

    # 5. Clean Mode Routing via computed config properties
    mode_index = config.mode_index

    if mode_index == 0:
        # Mode 0: Single Run
        script_content = build_single_mode(ctx, config)
        if script_content is None:
            return None, None
        return write_script(
            script_content, "single", exp_name, timestamp, backup_dir, config_file
        ), backup_dir

    elif mode_index == 1:
        # Mode 1: Inner Loop Only
        script_content = build_inner_loop_mode(ctx, config)
        if script_content is None:
            return None, None
        return write_script(
            script_content, "inner", exp_name, timestamp, backup_dir, config_file
        ), backup_dir

    else:
        # Mode 2: Array Mode (Outer + Inner Loops)
        script_content, total_tasks = build_job_array_mode(ctx, config, max_concurrent)
        if script_content is None:
            return None, None
        script_file = write_script(
            script_content, "array", exp_name, timestamp, backup_dir, config_file
        )
        print(f"({total_tasks} tasks)")
        return script_file, backup_dir

def extract_config_flags(config: AppConfig) -> Tuple[Set[str], Set[str]]:
    """Collects all argument keys across all configuration sections in AppConfig
    and converts them to CLI flags.
    """
    raw_keys: Set[str] = set()

    # 1. Static arguments
    if config.args:
        raw_keys.update(config.args.keys())

    # 2. Inner loop argument names
    inner_cfg = config.inner_loop
    if inner_cfg is not None:
        if isinstance(inner_cfg, TabularInnerLoop):
            # arg_name_list is populated from either 'args' or the 'arg_names' shorthand
            raw_keys.update(inner_cfg.arg_name_list)
        elif getattr(inner_cfg, "arg_name", None):
            raw_keys.add(inner_cfg.arg_name)

    # 3. Outer loop argument names
    for block in config.outer_loops:
        if isinstance(block, TabularOuterLoop):
            for arg_spec in block.args:
                raw_keys.add(arg_spec.arg_name)
        elif hasattr(block, "arg_name") and getattr(block, "arg_name"):
            raw_keys.add(getattr(block, "arg_name"))
        elif hasattr(block, "arg_names") and getattr(block, "arg_names"):
            raw_keys.update(getattr(block, "arg_names"))

    # 4. Experiment tracking arguments
    if config.experiment and config.experiment.tracking_args:
        raw_keys.update(config.experiment.tracking_args.keys())

    # Convert keys into CLI flag formats (e.g., "lr" -> "--lr")
    formatted_flags: Set[str] = set()
    for key in raw_keys:
        key_str = str(key).strip()
        flag = key_str if key_str.startswith("-") else f"--{key_str}"
        formatted_flags.add(flag)

    return formatted_flags, raw_keys


def validate_script_args(config: AppConfig) -> Tuple[bool, List[str]]:
    """Validates if arguments defined in the AppConfig instance are supported
    by the target executable via --help.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_unsupported_flags)
    """
    exec_cfg = config.execution
    interpreter = exec_cfg.interpreter
    executable_path = str(exec_cfg.executable)
    modules = exec_cfg.modules

    # Collect flags planned across all modes
    flags_to_check, _ = extract_config_flags(config)
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

    supported_flags: Set[str] = set()
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

def submit_slurm_script(
    script_path: Path,
    config: AppConfig,
    checkargs_q: bool,
    backup_dir: Optional[Path] = None,
) -> None:
    """Submits the generated script to SLURM.

    If submission was attempted but produced no job - a rejected sbatch, a
    missing sbatch, or arguments that failed validation before sbatch ran - the
    run's backup directory is discarded, so only snapshots corresponding to a
    real job survive. A run generated without --submit keeps its snapshot, since
    it is still waiting to be submitted by hand.
    """

    if checkargs_q:
        print("\n[INFO] Validating arguments passed to executable...")
        print("[INFO] This may take some time. Use --noargcheck to disable argument checking.")
        print('[INFO] CAUTION: Validation function expects all args in "--<args>" format!')
        print("[INFO] This includes single letter args!")
        
        validation_result, failed_args = validate_script_args(config)
        if not validation_result:
            print("\n[ERROR] Argument validation failed!", file=sys.stderr)
            print(f"[ERROR] Offending args: {' '.join(failed_args)}", file=sys.stderr)
            discard_backup_dir(backup_dir)
            sys.exit(1)
        else:
            print("\n[SUCCESS] Argument validation succeeded!")
    
    print(f"\n[INFO] Submitting {script_path} to SLURM...")
    try:
        # Calls sbatch. The embedded checkpoint logic handles skips securely.
        result = subprocess.run(["sbatch", str(script_path)], check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] SLURM Submission failed:\n{e.stderr}", file=sys.stderr)
        discard_backup_dir(backup_dir)
        sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] 'sbatch' command not found. Are you on a SLURM cluster node?", file=sys.stderr)
        discard_backup_dir(backup_dir)
        sys.exit(1)

# Collection function from https://share.gemini.google/ldX5zTud8zOv, https://share.gemini.google/IEM7GOQgmFCx
# Note that this function is specific to the NLODiffraction project. A more general collection function needs
# to be developed.
def collect_slurm_results_to_db(config_path: str, job_id: Optional[str]=None) -> None:
    """Collects SLURM job execution results and metadata into a SQLite database.

    Args:
        config_path (str): Path to the orchestrator JSON configuration file.
        job_id (Optional[str]): Specific job ID to collect results for. If None, collects all jobs.
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
    if job_id is not None:
        db_file_path = os.path.join(exp_dir, f"results_{job_id}.db")

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
        elif job_id and job_id not in filename:
            # If job_id is specified, skip files that don't correspond to it
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
        "config_paths",
        nargs="*",
        default=["config.json"],
        help="Path to the JSON configuration file(s) (default: config.json)"
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
        "--notimecheck",
        action="store_true",
        help="Disable checking the requested time limit against the partition's MaxTime."
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collects results from finished SLURM jobs."
    )
    parser.add_argument(
            "--collect-job",
            default=None,
            help="Collects results from finished SLURM jobs."
        )
    args = parser.parse_args()

    checkargsQ = not args.noargcheck

    for config_path in args.config_paths:
        print(f"\n--- Processing: {config_path} ---")

        if args.collect:
            collect_slurm_results_to_db(config_path)
            continue
        if args.collect_job is not None:
            collect_slurm_results_to_db(config_path, job_id=args.collect_job)
            continue

        generated_script, backup_dir = generate_slurm_script(
            config_path, args.dryrun, checktimeQ=not args.notimecheck
        )

        if generated_script is None:
            print(f"[INFO] Nothing to submit for {config_path}: all tasks already checkpointed.")
            continue

        if args.dryrun:
            print(f"[INFO] Dry-run: not submitting to SLURM for {config_path}.")
        elif args.submit:
            cfg = get_dict_from_config_file(config_path)
            config: AppConfig = validate_config(cfg, config_path, args.dryrun)
            submit_slurm_script(generated_script, config, checkargsQ, backup_dir)
        else:
            print(f"[TIP] Run 'sbatch {generated_script}' to manually submit, or pass --submit next time.")
