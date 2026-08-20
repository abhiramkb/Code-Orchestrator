from pathlib import Path
import itertools
from typing import Any, Dict, List, Optional, Tuple

from config_schema import (
    AppConfig,
    ExecutionConfig,
    ExplicitOuterLoop,
    InnerLoopConfig,
    OuterLoopBlock,
    RangeOuterLoop,
    TabularOuterLoop,
)

def build_exec_command(
    exec_cfg: ExecutionConfig,
    flags_str: str,
    fixed_args_str: str,
    extra_args_str: str,
) -> str:
    """Constructs the full execution command string using ExecutionConfig model attributes."""
    cmd_parts = [
        exec_cfg.interpreter,
        flags_str,
        str(exec_cfg.executable),
        fixed_args_str,
        extra_args_str,
    ]

    return " ".join(p for p in cmd_parts if p)


def build_single_mode(ctx: Dict[str, Any], config: AppConfig) -> str:
    """Mode 0: Non-looped execution script generator."""
    exec_cmd = build_exec_command(
        ctx["exec_cfg"], ctx["flags_str"], ctx["fixed_args_str"], ""
    )
    return f"""{ctx['common_header']}

# Navigate to executable directory to preserve repo/git context
cd "{ctx['exec_dir']}" || exit 1

indicator="SINGLE"

EXEC_SIG="{ctx['exec_sig_str']}"
exec_hash=$(printf '%s' "$EXEC_SIG" | md5sum | cut -d ' ' -f 1)

# Checkpoint check
CHECKPOINT_FILE="$CHECKPOINT_DIR/${{indicator}}_${{exec_hash}}.done"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "[CHECKPOINT] Configuration previously completed. Exiting."
    exit 0
fi

echo "======================= SLURM RUN LEGEND ======================="
echo "Mode: Single Run (loopQ = false)"
echo "Working Directory: {ctx['exec_dir']}"
echo "Exec Path: {ctx['exec_path']}"
echo "Flags: {ctx['fixed_args_str']}"
echo "================================================================"

indicator="SINGLE"
{ctx['exp_args_block']}

print_args "$indicator" "{ctx['fixed_args_str']} $exp_args"

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

def _build_inner_arg_mapping(inner_cfg: InnerLoopConfig) -> str:
    """Generates Bash lines that map array indices to CLI flags."""
    mapping_lines = []
    for spec in inner_cfg.args:
        bash_val = f"${{inner_vals[{spec.column}]}}"
        if spec.template:
            # Resolves Python template to Bash variable (e.g., 'val_{val}' -> 'val_${inner_vals[0]}')
            bash_val = spec.template.format(val=bash_val)
        mapping_lines.append(f'    inner_args_str+=" --{spec.arg_name} {bash_val}"')
    return "\n".join(mapping_lines)

def build_inner_loop_mode(ctx: Dict[str, Any], config: AppConfig) -> str:
    """Mode 1: Inner-loop-only script generator."""
    
    inner_cfg = config.inner_loop

    if inner_cfg.file_path is None:
        raise ValueError("InnerLoopConfig.file_path cannot be None for Inner Loop Mode.")

    target_inner_file = str(inner_cfg.file_path.resolve())

    comment_prefix = inner_cfg.comment_prefix
    exec_cmd = build_exec_command(
        ctx["exec_cfg"], ctx["flags_str"], ctx["fixed_args_str"], "$inner_args_str"
    )

    # Dynamic Bash mapping for indexed columns
    arg_mapping = _build_inner_arg_mapping(inner_cfg)
    arg_names_display = ", ".join(inner_cfg.arg_name_list)

    #print("arg_names_display:", arg_names_display)
    #exit()

    # Set Bash IFS for custom delimiters (defaults to space)
    ifs_setting = f"IFS='{inner_cfg.delimiter}' " if inner_cfg.delimiter != " " else ""


    return f"""{ctx['common_header']}

# Navigate to executable directory to preserve repo/git context
cd "{ctx['exec_dir']}" || exit 1

EXEC_SIG="{ctx['exec_sig_str']}"

echo "======================= SLURM RUN LEGEND ======================="
echo "Mode: Inner Loop Only"
echo "Working Directory: {ctx['exec_dir']}"
echo "Exec Path: {ctx['exec_path']}"
echo "Fixed Args: {ctx['fixed_args_str']}"
echo "Inner Loop Source File: {target_inner_file}"
echo "Inner Args: {arg_names_display}"
echo "================================================================"

line_no=0
while read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" == "{comment_prefix}"* ]] && continue
    ((line_no++))

    line_hash=$(printf '%s' "$EXEC_SIG $line" | md5sum | cut -d ' ' -f 1)
    indicator="L${{line_no}}"
    indicator_checkpoint="L${{line_no}}_${{line_hash}}"

    CHECKPOINT_FILE="$CHECKPOINT_DIR/${{indicator_checkpoint}}.done"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "[CHECKPOINT] Skipping $indicator - already completed."
        echo "[CHECKPOINT] Hash: ${{line_hash}}"
        continue
    fi

    {ifs_setting}read -r -a inner_vals <<< "$line"
    inner_args_str=""
{arg_mapping}
{ctx['exp_args_block']}

    print_args "$indicator" "{ctx['fixed_args_str']} $inner_args_str $exp_args"

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

def _evaluate_outer_block(block: OuterLoopBlock) -> List[Dict[str, Any]]:
    """Helper to evaluate a typed outer loop Pydantic block into argument dicts."""
    if isinstance(block, ExplicitOuterLoop):
        return [{block.arg_name: val} for val in block.values]

    elif isinstance(block, RangeOuterLoop):
        arg_name = block.arg_name
        start, stop, step = block.start, block.stop, block.step
        values = []

        if any(isinstance(x, float) for x in (start, stop, step)):
            curr = start
            while (step > 0 and curr < stop) or (step < 0 and curr > stop):
                values.append(round(curr, 10))
                curr += step
        else:
            values = list(range(int(start), int(stop), int(step)))

        return [{arg_name: val} for val in values]

    elif isinstance(block, TabularOuterLoop):
        rows = []
        with open(block.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if block.skip_blank_lines and not line_str:
                    continue
                if block.comment_prefix and line_str.startswith(block.comment_prefix):
                    continue

                if block.delimiter and block.delimiter != " ":
                    cols = [c.strip() for c in line_str.split(block.delimiter)]
                else:
                    cols = line_str.split()

                row_dict = {}
                for arg_spec in block.args:
                    raw_val = cols[arg_spec.column]
                    if arg_spec.template:
                        val = arg_spec.template.format(val=raw_val)
                    else:
                        val = raw_val
                    row_dict[arg_spec.arg_name] = val

                rows.append(row_dict)
        return rows

    raise ValueError(f"Unsupported outer_loop block type: '{type(block)}'")

# Outer array with increase flexibility: https://share.gemini.google/pVLNoK7EKGXq
def build_job_array_mode(
    ctx: Dict[str, Any], config: AppConfig,
    max_concurrent: Optional[int] = None,
) -> Tuple[str, int]:
    """Mode 2: Outer + Inner loop SLURM Job Array script generator."""
    # 1. Evaluate each typed Pydantic block into a list of argument dictionaries
    inner_cfg = config.inner_loop
    outer_cfg = config.outer_loops
    evaluated_blocks = [_evaluate_outer_block(block) for block in outer_cfg]

    comment_prefix = inner_cfg.comment_prefix

    # 2. Compute the Cartesian product across all outer loop blocks and merge
    outer_combinations = []
    for combo_tuple in itertools.product(*evaluated_blocks):
        merged_combo = {}
        for d in combo_tuple:
            merged_combo.update(d)
        outer_combinations.append(merged_combo)

    total_tasks = len(outer_combinations)

    array_range = f"0-{total_tasks - 1}"
    if max_concurrent:
        array_range += f"%{max_concurrent}"

    # Inject array SBATCH directive after #!/bin/bash
    lines = ctx["common_header"].splitlines()
    lines.insert(1, f"#SBATCH --array={array_range}")
    header_with_array = "\n".join(lines)

    bash_outer_args, bash_outer_descs, bash_target_files = [], [], []

    for combo_dict in outer_combinations:
        outer_args_str = " ".join(f"--{k} {v}" for k, v in combo_dict.items())
        outer_desc = ", ".join(f"{k}={v}" for k, v in combo_dict.items())

        target_inner_file = inner_cfg.file_path
        if target_inner_file:
            target_inner_file_str = str(target_inner_file.resolve())
        else:
            target_inner_file_str = ""
            for loop_cfg in outer_cfg:
                arg_name = getattr(loop_cfg, "arg_name", None)
                val = combo_dict.get(arg_name) if arg_name else None
                inner_files = getattr(loop_cfg, "inner_files", None)
                if inner_files and val in inner_files:
                    target_inner_file_str = str(Path(inner_files[val]).resolve())
                    break

        bash_outer_args.append(f'    "{outer_args_str}"')
        bash_outer_descs.append(f'    "{outer_desc}"')
        bash_target_files.append(f'    "{target_inner_file_str}"')

    args_array_block = "\n".join(bash_outer_args)
    descs_array_block = "\n".join(bash_outer_descs)
    files_array_block = "\n".join(bash_target_files)

    exec_cmd = build_exec_command(
        ctx["exec_cfg"],
        ctx["flags_str"],
        ctx["fixed_args_str"],
        "$outer_args_str $inner_args_str",
    )

    # Dynamic Bash mapping for indexed columns
    arg_mapping = _build_inner_arg_mapping(inner_cfg)
    arg_names_display = ", ".join(inner_cfg.arg_name_list)

    # Set Bash IFS for custom delimiters (defaults to space)
    ifs_setting = f"IFS='{inner_cfg.delimiter}' " if inner_cfg.delimiter != " " else ""

    script_content = f"""{header_with_array}

# Navigate to executable directory to preserve repo/git context
cd "{ctx['exec_dir']}" || exit 1

EXEC_SIG="{ctx['exec_sig_str']}"

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
echo "Working Directory: {ctx['exec_dir']}"
echo "Exec Path: {ctx['exec_path']}"
echo "Fixed Args: {ctx['fixed_args_str']}"
echo "Outer Loop Combo: $outer_desc"
echo "Outer Flags: $outer_args_str"
echo "Inner Loop Source File: $target_inner_file"
echo "Inner Args: {arg_names_display}"
echo "======================================================================"

line_no=0
while read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" == "{comment_prefix}"* ]] && continue
    ((line_no++))

    combo_hash=$(printf '%s' "$EXEC_SIG $outer_args_str $line" | md5sum | cut -d ' ' -f 1)
    indicator="A${{TASK_ID}}_L${{line_no}}"
    indicator_checkpoint="A${{TASK_ID}}_L${{line_no}}_${{combo_hash}}"

    CHECKPOINT_FILE="$CHECKPOINT_DIR/${{indicator_checkpoint}}.done"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "[CHECKPOINT] Skipping $indicator - already completed."
        echo "[CHECKPOINT] Hash: ${{combo_hash}}"
        continue
    fi

    {ifs_setting}read -r -a inner_vals <<< "$line"
    inner_args_str=""
{arg_mapping}
{ctx['exp_args_block']}

    print_args "$indicator" "{ctx['fixed_args_str']} $outer_args_str $inner_args_str $exp_args"

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
    return script_content, total_tasks