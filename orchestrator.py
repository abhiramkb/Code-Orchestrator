import json
from datetime import datetime
import shutil
import itertools
import argparse
import sys
from pathlib import Path
import subprocess
import re

def determine_loop_q(cfg: dict) -> bool:
    """Determines if loop mode is active based on config flags and structure."""
    if "loopQ" in cfg:
        return bool(cfg["loopQ"])
    
    # If loopQ is missing, check if loop options are absent while args is present
    has_inner = "inner_loop" in cfg
    has_outer = "outer_loops" in cfg
    has_args = "args" in cfg
    has_slurm = "slurm" in cfg
    has_execution = "execution" in cfg

    if not has_inner and not has_outer and has_args and has_slurm and has_execution:
        return False

    return True


def validate_config(cfg: dict, config_path: str):
    """Validates structure, data types, and file existence for the configuration dictionary."""
    errors = []

    # 1. Validate top-level keys
    loop_q = determine_loop_q(cfg)
    
    if "args" in cfg and not isinstance(cfg["args"], dict):
        errors.append("'args' must be an object (key-value pairs).")

    if "execution" not in cfg or not isinstance(cfg["execution"], dict):
        errors.append("Missing or invalid 'execution' section (must be an object).")
    else:
        exec_cfg = cfg["execution"]
        if "language" not in exec_cfg or not isinstance(exec_cfg["language"], str):
            errors.append("Missing or invalid 'execution.language' (must be a string).")
        if "executable" not in exec_cfg or not isinstance(exec_cfg["executable"], str):
            errors.append("Missing or invalid 'execution.executable' (must be a string).")
        if "flags" in exec_cfg and not isinstance(exec_cfg["flags"], list):
            errors.append("'execution.flags' must be an array of strings.")
        if "modules" in exec_cfg and not isinstance(exec_cfg["modules"], list):
            errors.append("'execution.modules' must be an array of strings.")
        if "env_vars" in exec_cfg and not isinstance(exec_cfg["env_vars"], dict):
            errors.append("'execution.env_vars' must be an object (key-value pairs).")

    if "slurm" not in cfg or not isinstance(cfg["slurm"], dict):
        errors.append("Missing or invalid 'slurm' section (must be an object).")

    # 2. Mode-Specific Validation
    if not loop_q:
        # Single Run Mode
        if "args" in cfg and not isinstance(cfg["args"], dict):
            errors.append("'args' must be an object when 'loopQ' is false.")
    else:
        # Loop / Parameter Scan Mode
        if "inner_loop" not in cfg or not isinstance(cfg["inner_loop"], dict):
            errors.append("Missing or invalid 'inner_loop' section (required when loopQ=true).")
        else:
            inner_cfg = cfg["inner_loop"]
            if "arg_names" not in inner_cfg or not isinstance(inner_cfg["arg_names"], list):
                errors.append("Missing or invalid 'inner_loop.arg_names' (must be an array).")

        outer_cfg = cfg.get("outer_loops", [])
        if not isinstance(outer_cfg, list):
            errors.append("'outer_loops' must be an array.")

        # 3. Inner Loop Data File Verification
        if "inner_loop" in cfg and isinstance(cfg["inner_loop"], dict):
            inner_cfg = cfg["inner_loop"]
            target_inner_file = inner_cfg.get("file_path")

            if target_inner_file:
                p = Path(target_inner_file)
                #if not p.is_file():
                #    errors.append(f"Specified 'inner_loop.file_path' does not exist: '{target_inner_file}'")
            elif outer_cfg:
                file_mapped = False
                for idx, o_loop in enumerate(outer_cfg):
                    if isinstance(o_loop, dict) and "inner_files" in o_loop:
                        file_mapped = True
                        inner_files_map = o_loop["inner_files"]
                        if not isinstance(inner_files_map, dict):
                            errors.append(f"'outer_loops[{idx}].inner_files' must be an object.")
                            continue
                if not file_mapped:
                    errors.append("No 'file_path' provided in 'inner_loop' and no 'inner_files' mappings found in 'outer_loops'.")
            else:
                errors.append("No 'file_path' specified in 'inner_loop' and no 'outer_loops' defined.")

    # 4. Experiment Tracking Validation
    if "experiment" in cfg:
        exp_cfg = cfg["experiment"]
        if not isinstance(exp_cfg, dict):
            errors.append("'experiment' section must be an object.")
        else:
            for req_key in ["result_database_path", "experiment_name", "slrm_output_dir"]:
                if req_key not in exp_cfg or not isinstance(exp_cfg[req_key], str):
                    errors.append(f"Missing or invalid 'experiment.{req_key}' (must be a string).")
            # Check for tracking_args dictionary
            if "tracking_args" not in exp_cfg or exp_cfg["tracking_args"] is None:
                print(
                    "[WARNING] 'tracking_args' is missing under 'experiment' in configuration.\n"
                    "          'exp_args' will default to an empty string.\n"
                    "          Example structure to enable dynamic tracking flags:\n"
                    "          \"experiment\": {\n"
                    "              \"tracking_args\": {\n"
                    "                  \"save_dir\": \"{save_dir}/{experiment_name}_{__indicator}\",\n"
                    "                  \"json\": \"{save_dir}/{experiment_name}_{__indicator}/result.json\"\n"
                    "              }\n"
                    "          }\n",
                    file=sys.stderr
                )
            elif not isinstance(exp_cfg["tracking_args"], dict):
                errors.append("'experiment.tracking_args' must be an object (key-value pairs).")

    if errors:
        print(f"\n[CONFIG ERROR] Validation failed for '{config_path}':", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


def generate_slurm_script(config_path, dryrunQ):
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

    # Perform pre-flight validation
    validate_config(cfg, config_path)
    print(f"[SUCCESS] Config validation passed for '{config_path}'.")

    loop_q = determine_loop_q(cfg)
    exec_cfg = cfg["execution"]
    slurm_cfg = cfg["slurm"].copy()
    experiment_cfg = cfg.get("experiment")

    # Resolve executable path and directory to be able to properly obtain git context
    exec_path = Path(exec_cfg["executable"]).resolve()
    exec_dir = exec_path.parent

    # Global fixed arguments passed in every run mode (looped/non-looped)
    fixed_args_cfg = cfg.get("args", {})
    fixed_args_list = []
    for k, v in fixed_args_cfg.items():
        if isinstance(v, str):
            fixed_args_list.append(f'--{k} \\"{v}\\"')
        else:
            fixed_args_list.append(f'--{k} {v}')
    fixed_args_str = " ".join(fixed_args_list)

    # Handle experiment tracking paths and SLURM header generation
    if not experiment_cfg:
        exp_name = "noexp"
    if experiment_cfg:
        db_path = experiment_cfg["result_database_path"]
        exp_name = experiment_cfg["experiment_name"]
        slrm_output_dir_name = experiment_cfg["slrm_output_dir"]
        
        # SLURM output directory path: <result_database_path>/<experiment_name>/<slrm_output_dir>
        full_slrm_output_dir = f"{db_path}/{exp_name}/{slrm_output_dir_name}"
        
        # Set SLURM standard output option: <full_slrm_output_dir>/<experiment_name>_%j.out
        slurm_cfg["output"] = f"{full_slrm_output_dir}/{exp_name}_%j.out"
        
        # Ensure target SLURM output directory exists prior to job submission
        try:
            Path(full_slrm_output_dir).mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Ensured SLURM output directory exists: {full_slrm_output_dir}")
        except Exception as e:
            print(f"[WARNING] Could not create SLRM_OUTPUT_DIR '{full_slrm_output_dir}': {e}", file=sys.stderr)

        # Copy configuration file to <result_database_path>/<experiment_name>/<datetime_string>/
        if not dryrunQ:
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(db_path) / exp_name / datetime_str
            try:
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_file, backup_dir / config_file.name)
                print(f"[INFO] Copied config file to: {backup_dir / config_file.name}")
            except Exception as e:
                print(f"[WARNING] Could not copy config file to '{backup_dir}': {e}", file=sys.stderr)

        experiment_env_block = f"""
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
        # Check for dynamic tracking arguments in config
        tracking_args = experiment_cfg.get("tracking_args")

        arg_components = []
        if tracking_args is not None:
          for key, val_template in tracking_args.items():
              formatted_val = val_template 
              # Translate placeholder keywords to bash variable syntaxes
              formatted_val = formatted_val.replace("{result_database_path}", "${RESULT_DATABASE_PATH}")  
              formatted_val = formatted_val.replace("{experiment_name}", "${EXPERIMENT_NAME}")
              formatted_val = formatted_val.replace("{slrm_output_dir}", "${SLRM_OUTPUT_DIR}")
              formatted_val = formatted_val.replace("{save_dir}", "${SAVE_DIR}")
              formatted_val = formatted_val.replace("{__indicator}", "${indicator}")
                  
              arg_components.append(f'--{key} \\"{formatted_val}\\"')

        tracking_args_str = " ".join(arg_components)

        exp_args_block = f"""
    exp_args=""
    if [ -n "${{SAVE_DIR:-}}" ]; then
        exp_args="{tracking_args_str}"
    fi"""
    else:
        # Fallback checkpoint directory if no experiment block is defined
        experiment_env_block = """
export CHECKPOINT_DIR="./checkpoints"
mkdir -p "$CHECKPOINT_DIR"
"""
        exp_args_block = '\n    exp_args=""'

    # Extract job array throttling if provided
    max_concurrent = slurm_cfg.pop("max_concurrent_tasks", None)

    # Build SLURM header options
    slurm_header = "".join(f'#SBATCH --{k}={v}\n' for k, v in slurm_cfg.items())

    # Environment block setup
    modules = exec_cfg.get("modules", [])
    module_load_block = (
        "\n# Load Required Environment Modules\n" + "\n".join(f"module load {m}" for m in modules)
        if modules else "# No environment modules specified"
    )

    env_vars = exec_cfg.get("env_vars", {})
    env_var_block = (
        "\n# Set Environment Variables\n" + "\n".join(f'export {k}={v}' for k, v in env_vars.items())
        if env_vars else "# No environment variables specified"
    )

    flags_str = " ".join(exec_cfg.get("flags", []))

    # Full execution signature string used for checkpoint hashing
    exec_sig_components = [exec_cfg['language'], flags_str, str(exec_path), fixed_args_str]
    exec_sig_str = " ".join(p for p in exec_sig_components if p)

    # =========================================================================
    # SINGLE RUN MODE (loopQ == False)
    # =========================================================================
    if not loop_q:
        #args_cfg = cfg.get("args", {})
        #args_str = " ".join(f"--{k} {v}" for k, v in args_cfg.items())
        #exec_cmd = f"{exec_cfg['language']} {flags_str} {exec_cfg['executable']} {args_str}".strip()
        cmd_parts = [exec_cfg['language'], flags_str, exec_cfg['executable'], fixed_args_str]
        exec_cmd = " ".join(p for p in cmd_parts if p)

        script_content = f"""#!/bin/bash
{slurm_header}
{experiment_env_block}
{module_load_block}
{env_var_block}

# Navigate to executable directory to preserve repo/git context
cd "{exec_dir}" || exit 1

indicator="SINGLE"

EXEC_SIG="{exec_sig_str}"
exec_hash=$(printf '%s' "$EXEC_SIG" | md5sum | cut -d ' ' -f 1)

# Checkpoint check
CHECKPOINT_FILE="$CHECKPOINT_DIR/${{indicator}}_${{exec_hash}}.done"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "[CHECKPOINT] Configuration previously completed. Exiting."
    exit 0
fi

echo "======================= SLURM RUN LEGEND ======================="
echo "Mode: Single Run (loopQ = false)"
echo "Working Directory: {exec_dir}"
echo "Exec Path: {exec_path}"
echo "Flags: {fixed_args_str}"
echo "================================================================"

indicator="SINGLE"
{exp_args_block}

{{
    starttime=$(date +%s%N)
    echo "Job started at: $(date)"
    
    eval "{exec_cmd} $exp_args"
    EXIT_CODE=$?
    
    endtime=$(date +%s%N)
    echo "Job finished at: $(date) with Exit Code: $EXIT_CODE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        touch "$CHECKPOINT_FILE"
    fi

    elapsedtime=$((endtime - starttime))
    sec=$(( elapsedtime / 1000000000 ))
    msec=$(( (elapsedtime % 1000000000) / 1000000 ))
    printf "Job duration: %d.%03d seconds\\n" "$sec" "$msec"
}} > >(sed "s/^/[${{indicator}}_out] /") 2> >(sed "s/^/[${{indicator}}_err] /" >&2)

wait
"""
        #script_file = Path("submit_single.sh")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_file = Path(f"submit_single_{exp_name}_{timestamp}.sh")
        script_file.write_text(script_content)
        print(f"Generated single-run SLURM script: {script_file}")
        return script_file

    # =========================================================================
    # INNER LOOP ONLY MODE (loopQ == True, outer_loops empty/omitted)
    # =========================================================================
    inner_cfg = cfg["inner_loop"]
    outer_cfg = cfg.get("outer_loops", [])

    if not outer_cfg:
        target_inner_file = str(Path(inner_cfg["file_path"]).resolve())
        #target_inner_file = inner_cfg["file_path"]
        #exec_cmd = f"{exec_cfg['language']} {flags_str} {exec_cfg['executable']} $inner_args_str".strip()
        # Add fixed_args_str before inner loop arguments
        cmd_parts = [exec_cfg['language'], flags_str, exec_cfg['executable'], fixed_args_str, "$inner_args_str"]
        exec_cmd = " ".join(p for p in cmd_parts if p)

        script_content = f"""#!/bin/bash
{slurm_header}
{experiment_env_block}
{module_load_block}
{env_var_block}

# Navigate to executable directory to preserve repo/git context
cd "{exec_dir}" || exit 1

EXEC_SIG="{exec_sig_str}"

echo "======================= SLURM RUN LEGEND ======================="
echo "Mode: Inner Loop Only"
echo "Working Directory: {exec_dir}"
echo "Exec Path: {exec_path}"
echo "Fixed Args: {fixed_args_str}"
echo "Inner Loop Source File: {target_inner_file}"
echo "Inner Args: {', '.join(inner_cfg['arg_names'])}"
echo "================================================================"

line_no=0
while read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    ((line_no++))

    # Generate an MD5 hash combining line's contents and execution signature
    line_hash=$(printf '%s' "$EXEC_SIG $line" | md5sum | cut -d ' ' -f 1)
    
    indicator="L${{line_no}}"

    # Append the hash to the indicator
    indicator_checkpoint="L${{line_no}}_${{line_hash}}"
    

    CHECKPOINT_FILE="$CHECKPOINT_DIR/${{indicator_checkpoint}}.done"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "[CHECKPOINT] Skipping $indicator - already completed."
        continue
    fi

    read -r -a inner_vals <<< "$line"
    inner_args_str=""
"""
        for c_idx, arg_name in enumerate(inner_cfg["arg_names"]):
            script_content += f'    inner_args_str+=" --{arg_name} ${{inner_vals[{c_idx}]}}"\n'

        script_content += f"""{exp_args_block}

    {{
        starttime=$(date +%s%N)
        echo "Job started at: $(date)"
        
        eval "{exec_cmd} $exp_args"
        EXIT_CODE=$?
        
        endtime=$(date +%s%N)
        echo "Job finished at: $(date) with Exit Code: $EXIT_CODE"

        if [ $EXIT_CODE -eq 0 ]; then
          touch "$CHECKPOINT_FILE"
        fi
        
        elapsedtime=$((endtime - starttime))
        sec=$(( elapsedtime / 1000000000 ))
        msec=$(( (elapsedtime % 1000000000) / 1000000 ))
        printf "Job duration: %d.%03d seconds\\n" "$sec" "$msec"
        echo ""
    }} > >(sed "s/^/[${{indicator}}_out] /") 2> >(sed "s/^/[${{indicator}}_err] /" >&2)

done < "{target_inner_file}"

wait
"""
        #script_file = Path("submit_inner.sh")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_file = Path(f"submit_inner_{exp_name}_{timestamp}.sh")
        script_file.write_text(script_content)
        print(f"Generated inner-loop SLURM script: {script_file}")
        return script_file

    # =========================================================================
    # JOB ARRAY MODE (loopQ == True with outer_loops)
    # =========================================================================
    outer_keys = [item["arg_name"] for item in outer_cfg]
    outer_val_lists = [item["values"] for item in outer_cfg]
    outer_combinations = list(itertools.product(*outer_val_lists))
    total_tasks = len(outer_combinations)

    array_range = f"0-{total_tasks - 1}"
    if max_concurrent:
        array_range += f"%{max_concurrent}"

    array_header = f"#SBATCH --array={array_range}\n"

    bash_outer_args = []
    bash_outer_descs = []
    bash_target_files = []

    for combo in outer_combinations:
        combo_dict = dict(zip(outer_keys, combo))
        outer_args_str = " ".join(f"--{k} {v}" for k, v in combo_dict.items())
        outer_desc = ", ".join(f"{k}={v}" for k, v in combo_dict.items())
        
        target_inner_file = inner_cfg.get("file_path")
        if target_inner_file:
            target_inner_file = str(Path(target_inner_file).resolve())
        elif outer_cfg:
            for loop_cfg in outer_cfg:
                arg_name = loop_cfg["arg_name"]
                val = combo_dict.get(arg_name)
                if "inner_files" in loop_cfg and val in loop_cfg["inner_files"]:
                    target_inner_file = loop_cfg["inner_files"][val]
                    break

        bash_outer_args.append(f'    "{outer_args_str}"')
        bash_outer_descs.append(f'    "{outer_desc}"')
        bash_target_files.append(f'    "{target_inner_file}"')

    args_array_block = "\n".join(bash_outer_args)
    descs_array_block = "\n".join(bash_outer_descs)
    files_array_block = "\n".join(bash_target_files)

    #exec_cmd = f"{exec_cfg['language']} {flags_str} {exec_cfg['executable']} $outer_args_str $inner_args_str".strip()
    # Add fixed_args_str before outer and inner loop arguments
    cmd_parts = [exec_cfg['language'], flags_str, exec_cfg['executable'], fixed_args_str, "$outer_args_str $inner_args_str"]
    exec_cmd = " ".join(p for p in cmd_parts if p)

    script_content = f"""#!/bin/bash
{slurm_header}{array_header}
{experiment_env_block}
{module_load_block}
{env_var_block}

# Navigate to executable directory to preserve repo/git context
cd "{exec_dir}" || exit 1

EXEC_SIG="{exec_sig_str}"

# Parameter mapping arrays built by orchestrator.py
OUTER_ARGS=(
{args_array_block}
)

OUTER_DESCS=(
{descs_array_block}
)

TARGET_FILES=(
{files_array_block}
)

# Safe task ID resolution (defaults to 0 if executed manually outside SLURM)
TASK_ID=${{SLURM_ARRAY_TASK_ID:-0}}

outer_args_str="${{OUTER_ARGS[$TASK_ID]}}"
outer_desc="${{OUTER_DESCS[$TASK_ID]}}"
target_inner_file="${{TARGET_FILES[$TASK_ID]}}"

echo "======================= SLURM ARRAY RUN LEGEND ======================="
echo "Array Task ID: $TASK_ID"
echo "Working Directory: {exec_dir}"
echo "Exec Path: {exec_path}"
echo "Fixed Args: {fixed_args_str}"
echo "Outer Loop Combo: $outer_desc"
echo "Outer Flags: $outer_args_str"
echo "Inner Loop Source File: $target_inner_file"
echo "Inner Args: {', '.join(inner_cfg['arg_names'])}"
echo "======================================================================"

line_no=0
while read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    ((line_no++))

    # Generate an MD5 hash of the inner line, outer arguments AND the execution signature
    combo_hash=$(printf '%s' "$EXEC_SIG $outer_args_str $line" | md5sum | cut -d ' ' -f 1)
    
    indicator="A${{TASK_ID}}_L${{line_no}}"

    indicator_checkpoint="A${{TASK_ID}}_L${{line_no}}_${{combo_hash}}"

    CHECKPOINT_FILE="$CHECKPOINT_DIR/${{indicator_checkpoint}}.done"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "[CHECKPOINT] Skipping $indicator - already completed."
        continue
    fi

    read -r -a inner_vals <<< "$line"
    inner_args_str=""
"""
    for c_idx, arg_name in enumerate(inner_cfg["arg_names"]):
        script_content += f'    inner_args_str+=" --{arg_name} ${{inner_vals[{c_idx}]}}"\n'

    script_content += f"""{exp_args_block}

    {{
        starttime=$(date +%s%N)
        echo "Job started at: $(date)"
        
        eval "{exec_cmd} $exp_args"
        EXIT_CODE=$?
        
        endtime=$(date +%s%N)
        echo "Job finished at: $(date) with Exit Code: $EXIT_CODE"

        if [ $EXIT_CODE -eq 0 ]; then
            touch "$CHECKPOINT_FILE"
        fi
        
        elapsedtime=$((endtime - starttime))
        sec=$(( elapsedtime / 1000000000 ))
        msec=$(( (elapsedtime % 1000000000) / 1000000 ))
        printf "Job duration: %d.%03d seconds\\n" "$sec" "$msec"
    }} > >(sed "s/^/[${{indicator}}_out] /") 2> >(sed "s/^/[${{indicator}}_err] /" >&2)

done < "$target_inner_file"

wait
"""

    #script_file = Path("submit_array.sh")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_file = Path(f"submit_array_{exp_name}_{timestamp}.sh")
    script_file.write_text(script_content)
    print(f"Generated SLURM Job Array script ({total_tasks} tasks): {script_file}")

    return script_file

def extract_config_flags(cfg: dict) -> set[str]:
    """Collects all argument keys across all configuration sections and converts them to CLI flags."""
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
            flag = f"-{key_str}" if len(key_str) == 1 else f"--{key_str}"
        else:
            flag = key_str
        formatted_flags.add(flag)

    print("formatted_flags = ",formatted_flags)

    return formatted_flags

def validate_script_args(config_path) -> tuple[bool, list[str]]:
    """Validates if arguments defined in the config dictionary are supported by the target executable via --help.

    Returns:
        tuple[bool, list[str]]: (is_valid, list_of_unsupported_flags)
    """
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
    
    exec_cfg = cfg.get("execution", {})
    interpreter = exec_cfg.get("language", "")
    executable_path = exec_cfg.get("executable", "")
    modules = exec_cfg.get("modules", [])

    if not interpreter or not executable_path:
        raise ValueError(
            "Missing required 'execution.language' or 'execution.executable' in config."
        )

    # Collect flags planned across all modes
    flags_to_check = extract_config_flags(cfg)
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

def submit_slurm_script(script_path: Path):
    """Submits the generated script to SLURM."""
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
        "--checkargs",
        action="store_true",
        help="Check if the args passed to the executable are supported."
    )
    args = parser.parse_args()
    
    generated_script = generate_slurm_script(args.config, args.dryrun)

    if args.checkargs:
        result = validate_script_args(args.config)
        print(result)
        exit()

    if args.dryrun:
        print(f"\n[INFO] Dry-run: not submitting to SLURM.")
    elif args.submit:
        submit_slurm_script(generated_script)
    else:
        print(f"\n[TIP] Run 'sbatch {generated_script}' to manually submit, or pass --submit next time.")