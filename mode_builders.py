from pathlib import Path
import itertools

def build_exec_command(exec_cfg: dict, flags_str: str, fixed_args_str: str, extra_args_str: str) -> str:
    """
    Constructs the full execution command string.
    extra_args_str can be something like "$inner_args_str" or "$outer_args_str $inner_args_str".
    """
    cmd_parts = [exec_cfg['language'], flags_str, exec_cfg['executable'], fixed_args_str, extra_args_str]
    return " ".join(p for p in cmd_parts if p)

def build_single_mode(ctx: dict) -> str:
    """Mode 1: Non-looped execution script generator."""
    exec_cmd = build_exec_command(ctx['exec_cfg'], ctx['flags_str'], ctx['fixed_args_str'], "")
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


def build_inner_loop_mode(ctx: dict, inner_cfg: dict) -> str:
    """Mode 2: Inner-loop-only script generator."""
    target_inner_file = str(Path(inner_cfg["file_path"]).resolve())
    exec_cmd = build_exec_command(ctx['exec_cfg'], ctx['flags_str'], ctx['fixed_args_str'], "$inner_args_str")

    arg_mapping = "\n".join(
        f'    inner_args_str+=" --{arg_name} ${{inner_vals[{idx}]}}"'
        for idx, arg_name in enumerate(inner_cfg["arg_names"])
    )

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
echo "Inner Args: {', '.join(inner_cfg['arg_names'])}"
echo "================================================================"

line_no=0
while read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
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

    read -r -a inner_vals <<< "$line"
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


def _evaluate_outer_block(block: dict) -> list[dict]:
    """Helper to evaluate an outer loop block into a list of argument dicts."""
    block_type = block.get("type", "explicit")

    if block_type == "explicit":
        arg_name = block["arg_name"]
        return [{arg_name: val} for val in block["values"]]

    elif block_type == "range":
        arg_name = block["arg_name"]
        start = block["start"]
        stop = block["stop"]
        step = block.get("step", 1)
        
        values = []
        curr = start
        if isinstance(step, float) or isinstance(start, float) or isinstance(stop, float):
            while (step > 0 and curr < stop) or (step < 0 and curr > stop):
                values.append(round(curr, 10))
                curr += step
        else:
            values = list(range(start, stop, step))
            
        return [{arg_name: val} for val in values]

    elif block_type == "tabular_file":
        file_path = block["file_path"]
        delimiter = block.get("delimiter")
        comment_prefix = block.get("comment_prefix", "#")
        skip_blank = block.get("skip_blank_lines", True)
        args_spec = block.get("args", [])

        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if skip_blank and not line_str:
                    continue
                if comment_prefix and line_str.startswith(comment_prefix):
                    continue

                if delimiter and delimiter != " ":
                    cols = [c.strip() for c in line_str.split(delimiter)]
                else:
                    cols = line_str.split()

                row_dict = {}
                for arg in args_spec:
                    raw_val = cols[arg["column"]]
                    if "template" in arg:
                        val = arg["template"].format(val=raw_val)
                    else:
                        val = raw_val
                    row_dict[arg["arg_name"]] = val

                rows.append(row_dict)
        return rows

    else:
        raise ValueError(f"Unsupported outer_loop block type: '{block_type}'")

# Outer array with increase flexibility: https://share.gemini.google/pVLNoK7EKGXq
def build_job_array_mode(ctx: dict, inner_cfg: dict, outer_cfg: list, max_concurrent: int) -> tuple[str, int]:
    """Mode 3: Outer + Inner loop SLURM Job Array script generator."""
    # 1. Evaluate each block into a list of argument dictionaries
    evaluated_blocks = [_evaluate_outer_block(block) for block in outer_cfg]

    # 2. Compute the Cartesian product across all outer loop blocks and merge dictionaries
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
    lines = ctx['common_header'].splitlines()
    lines.insert(1, f"#SBATCH --array={array_range}")
    header_with_array = "\n".join(lines)

    bash_outer_args, bash_outer_descs, bash_target_files = [], [], []

    for combo_dict in outer_combinations:
        outer_args_str = " ".join(f"--{k} {v}" for k, v in combo_dict.items())
        outer_desc = ", ".join(f"{k}={v}" for k, v in combo_dict.items())

        target_inner_file = inner_cfg.get("file_path")
        if target_inner_file:
            target_inner_file = str(Path(target_inner_file).resolve())
        else:
            for loop_cfg in outer_cfg:
                arg_name = loop_cfg.get("arg_name")
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

    exec_cmd = build_exec_command(ctx['exec_cfg'], ctx['flags_str'], ctx['fixed_args_str'], "$outer_args_str $inner_args_str")

    arg_mapping = "\n".join(
        f'    inner_args_str+=" --{arg_name} ${{inner_vals[{idx}]}}"'
        for idx, arg_name in enumerate(inner_cfg["arg_names"])
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
echo "Inner Args: {', '.join(inner_cfg['arg_names'])}"
echo "======================================================================"

line_no=0
while read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
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

    read -r -a inner_vals <<< "$line"
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