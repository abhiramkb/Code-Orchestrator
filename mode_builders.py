from pathlib import Path
import itertools
from typing import Any, Dict, List, Optional, Tuple
from arg_transform import _apply_transform
from utils import format_cli_args, strip_hyphens

from config_schema import (
    AppConfig,
    ExecutionConfig,
    ExplicitLoop,
    RangeLoop,
    TabularInnerLoop,
    TabularOuterLoop,
    InnerLoop,
    OuterLoopBlock
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

def _evaluate_inner_loop(loop: InnerLoop) -> List[Dict[str, Any]]:
    """Evaluates a typed inner loop into argument dictionaries."""
    if isinstance(loop, ExplicitLoop):
        return [{loop.arg_name: val} for val in loop.values]

    elif isinstance(loop, RangeLoop):
        arg_name = loop.arg_name
        start, stop, step = loop.start, loop.stop, loop.step
        values = []

        if any(isinstance(x, float) for x in (start, stop, step)):
            curr = start
            while (step > 0 and curr < stop) or (step < 0 and curr > stop):
                values.append(round(curr, 10))
                curr += step
        else:
            values = list(range(int(start), int(stop), int(step)))

        return [{arg_name: val} for val in values]

    elif isinstance(loop, TabularInnerLoop):
        file_path = loop.file_path
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if loop.skip_blank_lines and not line_str:
                    continue
                if loop.comment_prefix and line_str.startswith(loop.comment_prefix):
                    continue

                if loop.delimiter and loop.delimiter != " ":
                    cols = [c.strip() for c in line_str.split(loop.delimiter)]
                else:
                    cols = line_str.split()

                row_dict = {}
                for arg_spec in loop.args:
                    val = cols[arg_spec.column]

                    # 1. Apply numeric transformation if requested
                    if arg_spec.transform:
                        val = _apply_transform(val, arg_spec.transform)

                    # 2. Apply string template if defined
                    if arg_spec.template:
                        val = arg_spec.template.format(val=val)

                    row_dict[arg_spec.arg_name] = val

                rows.append(row_dict)
        return rows

    raise ValueError(f"Unsupported inner_loop type: '{type(loop)}'")

def build_inner_loop_mode(ctx: Dict[str, Any], config: AppConfig) -> str:
    """Mode 1: Inner-loop-only script generator."""
    evaluated_inner = _evaluate_inner_loop(config.inner_loop)


    bash_inner_args = []
    for item_dict in evaluated_inner:
        args_str = format_cli_args(item_dict)
        bash_inner_args.append(f'    "{args_str}"')

    args_array_block = "\n".join(bash_inner_args)

    arg_names = list(evaluated_inner[0].keys()) if evaluated_inner else []
    arg_names_display = ", ".join(arg_names)

    exec_cmd = build_exec_command(
        ctx["exec_cfg"], ctx["flags_str"], ctx["fixed_args_str"], "$inner_args_str"
    )

    return f"""{ctx['common_header']}

# Navigate to executable directory to preserve repo/git context
cd "{ctx['exec_dir']}" || exit 1

EXEC_SIG="{ctx['exec_sig_str']}"

INNER_ARGS=(
{args_array_block}
)

echo "======================= SLURM RUN LEGEND ======================="
echo "Mode: Inner Loop Only"
echo "Working Directory: {ctx['exec_dir']}"
echo "Exec Path: {ctx['exec_path']}"
echo "Fixed Args: {ctx['fixed_args_str']}"
echo "Inner Args: {arg_names_display}"
echo "================================================================"

line_no=0
for inner_args_str in "${{INNER_ARGS[@]}}"; do
    ((line_no++))

    line_hash=$(printf '%s' "$EXEC_SIG $inner_args_str" | md5sum | cut -d ' ' -f 1)
    indicator="L${{line_no}}"
    indicator_checkpoint="L${{line_no}}_${{line_hash}}"

    CHECKPOINT_FILE="$CHECKPOINT_DIR/${{indicator_checkpoint}}.done"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "[CHECKPOINT] Skipping $indicator - already completed."
        echo "[CHECKPOINT] Hash: ${{line_hash}}"
        continue
    fi


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

done

wait
"""

def _evaluate_outer_block(block: OuterLoopBlock) -> List[Dict[str, Any]]:
    """Evaluates a typed outer loop block into argument dictionaries."""
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
                    val = cols[arg_spec.column]

                    # 1. Apply numeric transformation if requested
                    if arg_spec.transform:
                        val = _apply_transform(val, arg_spec.transform)

                    # 2. Apply string template if defined
                    if arg_spec.template:
                        val = arg_spec.template.format(val=val)

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

    bash_outer_args, bash_outer_descs, bash_inner_args_sets = [], [], []
    all_inner_arg_names = set()

    for combo_dict in outer_combinations:
        outer_args_str = format_cli_args(combo_dict)
        outer_desc = ", ".join(f"{strip_hyphens(k)}={v}" for k, v in combo_dict.items())

         # Resolve target inner file if inner_cfg uses a dynamic file mapped from outer loops
        target_inner_file = getattr(inner_cfg, "file_path", None)
        target_inner_file_path = None
        if target_inner_file:
            target_inner_file_path = Path(target_inner_file).resolve()
        else:
            target_inner_file_str = ""
            for loop_cfg in outer_cfg:
                arg_name = getattr(loop_cfg, "arg_name", None)
                val = combo_dict.get(arg_name) if arg_name else None
                inner_files = getattr(loop_cfg, "inner_files", None)
                if inner_files and val in inner_files:
                    target_inner_file_path = Path(inner_files[val]).resolve()
                    break

        evaluated_inner = _evaluate_inner_loop(inner_cfg, override_file_path=target_inner_file_path)

        formatted_inner_lines = []
        for item in evaluated_inner:
            formatted_inner_lines.append(format_cli_args(item))
            all_inner_arg_names.update(item.keys())

        inner_set_str = "\n".join(formatted_inner_lines)

        bash_outer_args.append(f'    "{outer_args_str}"')
        bash_outer_descs.append(f'    "{outer_desc}"')
        bash_inner_args_sets.append(f'    "{inner_set_str}"')

    args_array_block = "\n".join(bash_outer_args)
    descs_array_block = "\n".join(bash_outer_descs)
    files_array_block = "\n".join(bash_target_files)

    exec_cmd = build_exec_command(
        ctx["exec_cfg"],
        ctx["flags_str"],
        ctx["fixed_args_str"],
        "$outer_args_str $inner_args_str",
    )

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

INNER_ARGS_SETS=(
{inner_sets_array_block}
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
echo "Inner Args: {arg_names_display}"
echo "======================================================================"

line_no=0
while IFS= read -r inner_args_str || [ -n "$inner_args_str" ]; do
    [[ -z "$inner_args_str" ]] && continue
    ((++line_no))

    combo_hash=$(printf '%s' "$EXEC_SIG $outer_args_str $inner_args_str" | md5sum | cut -d ' ' -f 1)
    indicator="A${{TASK_ID}}_L${{line_no}}"
    indicator_checkpoint="A${{TASK_ID}}_L${{line_no}}_${{combo_hash}}"

    CHECKPOINT_FILE="$CHECKPOINT_DIR/${{indicator_checkpoint}}.done"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "[CHECKPOINT] Skipping $indicator - already completed."
        echo "[CHECKPOINT] Hash: ${{combo_hash}}"
        continue
    fi

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

done <<< "$inner_args_raw"

wait
"""
    return script_content, total_tasks
