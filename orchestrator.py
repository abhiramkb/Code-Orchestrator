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

def determine_loop_q(cfg: dict) -> bool:
    """Determines if loop mode is active based on config flags and structure."""

    mode_index = 0 # 0 -> no loops, 1 -> inner loop only, 2 -> inner and outer loops
    
    if "loopQ" in cfg:
        mode_index = 1
        if "outer_loops" in cfg:
            mode_index = 2
        return True, mode_index
    
    # If loopQ is missing, check if loop options are absent while args is present
    has_inner = "inner_loop" in cfg
    has_outer = "outer_loops" in cfg
    has_args = "args" in cfg
    has_slurm = "slurm" in cfg
    has_execution = "execution" in cfg

    if not has_inner and not has_outer and has_args and has_slurm and has_execution:
        return False, mode_index

    if has_inner:
        mode_index = 1
        if has_outer:
            mode_index = 2

    return True, mode_index


def validate_config(cfg: dict, config_path: str):
    """Validates structure, data types, and file existence for the configuration dictionary."""
    errors = []

    # 1. Validate top-level keys
    (loop_q, mode_index) = determine_loop_q(cfg)
    
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
    return mode_index

def generate_slurm_script(config_path, dryrunQ):
    cfg = get_dict_from_config_file(config_path)
    config_file = Path(config_path) # Storing config file path in case it is needed
    
    # Perform pre-flight validation
    mode_index = validate_config(cfg, config_path)
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

    # Helper Bash function definition to parse and print arguments with indicator prefix
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
        cmd_parts = [exec_cfg['language'], flags_str, exec_cfg['executable'], fixed_args_str]
        exec_cmd = " ".join(p for p in cmd_parts if p)

        script_content = f"""#!/bin/bash
{slurm_header}
{experiment_env_block}
{module_load_block}
{env_var_block}
{print_args_def}

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

print_args "$indicator" "{fixed_args_str} $exp_args"

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
        # Add fixed_args_str before inner loop arguments
        cmd_parts = [exec_cfg['language'], flags_str, exec_cfg['executable'], fixed_args_str, "$inner_args_str"]
        exec_cmd = " ".join(p for p in cmd_parts if p)

        script_content = f"""#!/bin/bash
{slurm_header}
{experiment_env_block}
{module_load_block}
{env_var_block}
{print_args_def}

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

    print_args "$indicator" "{fixed_args_str} $inner_args_str $exp_args"

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
{print_args_def}

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

    print_args "$indicator" "{fixed_args_str} $outer_args_str $inner_args_str $exp_args"

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_file = Path(f"submit_array_{exp_name}_{timestamp}.sh")
    script_file.write_text(script_content)
    print(f"Generated SLURM Job Array script ({total_tasks} tasks): {script_file}")

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

    # 6. Prepare unique dynamic columns across all result files
    all_json_keys = []
    for rec in records:
        for key in rec["json_data"].keys():
            if key not in all_json_keys:
                all_json_keys.append(key)

    # First column: job_id | Middle: json fields | Last: job_duration
    column_names = ["job_id"] + all_json_keys + ["job_duration"]

    # 7. Create/Update SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    col_definitions = ['"job_id" TEXT PRIMARY KEY']
    for k in all_json_keys:
        col_definitions.append(f'"{k}" TEXT')
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