from pathlib import Path
import itertools
from textwrap import indent
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

from checkpoint_utils import (
    is_single_mode_completed,
    filter_inner_loop_tasks,
    get_pending_array_task_ids,
    format_slurm_array_range,
)

# ---------------------------------------------------------------------------
# Emitted bash templates.
#
# These are deliberately plain (non-f) raw strings, substituted via .replace()
# on @@TOKEN@@ placeholders. Do NOT convert them to f-strings: the bash below is
# dense with ${...} and $(( )) and every brace would need doubling, which fails
# only at job start - long after the script has been queued.
# ---------------------------------------------------------------------------

RUN_INNER_TASK_TMPL = r'''
# =========================================================================
# INNER TASK RUNNER
# Runs one inner-loop point: checkpoint marker, execution, timing.
#
# NOTE: the { ... } > >(sed ...) block and its trailing `wait` MUST stay inside
# this function body. At script top level that exact construct hangs forever
# (verified on bash 5.1), and without the `wait` the sed output is silently lost.
# =========================================================================
run_inner_task() {
    local line_no="$1"
    local inner_args_str="$2"
    local outer_args_str="$3"
    local cpu_list="$4"
    local CHECKPOINT_FILE="$5"
    local indicator="@@INDICATOR_PREFIX@@L${line_no}"
    local EXIT_CODE=0
    local exp_args starttime endtime elapsedtime sec msec

    # Confine this run (and every child it spawns) to its own slice of the
    # allocation, so concurrent runs do not all pin onto the same cores.
    if [ -n "$cpu_list" ] && command -v taskset >/dev/null 2>&1; then
        taskset -cp "$cpu_list" $BASHPID >/dev/null 2>&1 || \
            echo "[WARN] taskset failed for $indicator" >&2
    fi
@@EXP_ARGS_BLOCK@@
    print_args "$indicator" "@@FIXED_ARGS@@ $outer_args_str $inner_args_str $exp_args"

    {
        starttime=$(date +%s%N)
        echo "Job started at: $(date)"
@@CONCURRENCY_INFO@@
        eval "@@EXEC_CMD@@ $exp_args"
        EXIT_CODE=$?

        endtime=$(date +%s%N)
        echo "Job finished at: $(date) with Exit Code: $EXIT_CODE"

        if [ $EXIT_CODE -eq 0 ]; then
            touch "$CHECKPOINT_FILE"
        fi

        elapsedtime=$((endtime - starttime))
        sec=$(( elapsedtime / 1000000000 ))
        msec=$(( (elapsedtime % 1000000000) / 1000000 ))
        printf "Job duration: %d.%03d seconds\n" "$sec" "$msec"
        echo ""
    } > >(sed -u "s/^/[${indicator}_out] /") 2> >(sed -u "s/^/[${indicator}_err] /" >&2)

    # MANDATORY: flushes both sed process substitutions before returning.
    wait
    return "$EXIT_CODE"
}
'''

# Optional wrapper placed around an inner-loop body when one array element owns
# several outer combinations. The indicator keys on $o_idx (the outer combination
# index), never on the array task id - see build_job_array_mode.
OUTER_WRAPPER_HEAD_TMPL = r'''
for o_idx in "${MY_OUTER_IDS[@]}"; do
    outer_args_str="${OUTER_ARGS[$o_idx]}"
    outer_desc="${OUTER_DESCS[$o_idx]}"
    echo "---------- outer combination $o_idx: $outer_desc ----------"
'''

OUTER_WRAPPER_TAIL_TMPL = r'''
done
'''

# Sequential driver - emitted when no multithreading_level is configured.
SEQUENTIAL_LOOP_TMPL = r'''
for idx in "${!INNER_ARGS[@]}"; do
    inner_args_str="${INNER_ARGS[$idx]}"
    line_no="${INNER_LINE_NOS[$idx]}"
    line_hash=$(printf '%s' "$EXEC_SIG${outer_args_str:+ $outer_args_str} $inner_args_str" | md5sum | cut -d ' ' -f 1)
    indicator="@@INDICATOR_PREFIX@@L${line_no}"
    CHECKPOINT_FILE="$CHECKPOINT_DIR/${indicator}_${line_hash}.done"

    # Runtime guard: protects a re-submitted or SLURM-requeued script
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "[CHECKPOINT] Skipping $indicator - already completed."
        echo "[CHECKPOINT] Hash: ${line_hash}"
        continue
    fi

    run_inner_task "$line_no" "$inner_args_str" "$outer_args_str" "" "$CHECKPOINT_FILE"
done
'''

# Drain for the sequential driver. Kept separate from the loop body so that, when
# an array element owns several outer combinations, it runs once at the very end
# rather than at every combination boundary.
SEQUENTIAL_TAIL_TMPL = r'''
wait
'''

# Concurrency runtime - emitted only when multithreading_level is configured.
PARALLEL_RUNTIME_TMPL = r'''
# =========================================================================
# INNER-LOOP CONCURRENCY RUNTIME
# =========================================================================
MULTITHREADING_LEVEL=@@MULTITHREADING_LEVEL@@
MAX_PARALLEL=@@MAX_PARALLEL@@
export MULTITHREADING_LEVEL

# The :-$(nproc) fallback is required: an unset SLURM_CPUS_PER_TASK would make
# bash arithmetic yield NJOBS=0 and the dispatcher would never launch anything.
NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
NJOBS=$(( NCPU / MULTITHREADING_LEVEL ))
if [ "$NJOBS" -gt "$MAX_PARALLEL" ]; then NJOBS=$MAX_PARALLEL; fi
if [ "$NJOBS" -lt 1 ]; then NJOBS=1; fi

# Per-run output is buffered here, then emitted as one contiguous block.
# Uses SLURM_JOB_ID (unique per array element), never JOB_ID, which in array
# mode is SLURM_ARRAY_JOB_ID and therefore shared by every task in the array.
LOG_BUF_DIR="${SLRM_OUTPUT_DIR:-$PWD}/.partial/${SLURM_JOB_ID:-$$}"
if [ "$NJOBS" -gt 1 ]; then mkdir -p "$LOG_BUF_DIR"; fi

echo "Inner-loop concurrency: $NJOBS concurrent run(s) x $MULTITHREADING_LEVEL thread(s) of $NCPU CPUs"

# Emit one run's buffered output as a single contiguous block. Only the parent
# shell ever calls this, so single-writer gives atomicity with no locking.
_dump_buffer() {
    local b="$1"
    [ -f "$b" ] || return 0
    if [ -s "$b" ]; then
        cat "$b"
        # Guarantee a trailing newline, so the next block's [..._out] prefix can
        # never concatenate onto an unterminated final line.
        if [ "$(tail -c 1 "$b" | wc -l)" -eq 0 ]; then echo; fi
    fi
    rm -f "$b"
    return 0
}

# Completed runs are already in the log; this only recovers output from runs
# still in flight when the job is interrupted (e.g. walltime).
_FLUSHED=0
_flush_partial() {
    [ "$_FLUSHED" -eq 1 ] && return 0
    _FLUSHED=1
    [ -d "$LOG_BUF_DIR" ] || return 0
    local b
    for b in "$LOG_BUF_DIR"/*.log; do
        [ -e "$b" ] || continue
        echo "[PARTIAL] ---- incomplete output for $(basename "$b" .log) (job interrupted) ----"
        cat "$b"
        rm -f "$b"
    done
    rmdir "$LOG_BUF_DIR" 2>/dev/null
    # Also drop the shared .partial parent, but only once it is empty.
    rmdir "$(dirname "$LOG_BUF_DIR")" 2>/dev/null
    return 0
}
# _FLUSHED makes this idempotent: `exit` in the TERM handler re-triggers EXIT.
trap '_flush_partial' EXIT
trap 'echo "[SIGNAL] SIGTERM received (walltime?); flushing partial logs."; _flush_partial; exit 143' TERM

# Slice this job's own affinity mask into disjoint per-slot CPU lists.
_ALL_CPUS=()
if [ "$NJOBS" -gt 1 ] && command -v taskset >/dev/null 2>&1; then
    _mask=$(taskset -cp $$ 2>/dev/null | sed 's/.*: //')
    IFS=',' read -ra _parts <<< "$_mask"
    for _r in "${_parts[@]}"; do
        case "$_r" in
            *-*) for _c in $(seq "${_r%-*}" "${_r#*-}"); do _ALL_CPUS+=("$_c"); done ;;
            "")  ;;
            *)   _ALL_CPUS+=("$_r") ;;
        esac
    done
fi

_slot_cpus() {
    local start=$(( $1 * MULTITHREADING_LEVEL ))
    if [ "${#_ALL_CPUS[@]}" -lt "$(( start + MULTITHREADING_LEVEL ))" ]; then
        echo ""
        return 0
    fi
    local IFS=','
    echo "${_ALL_CPUS[*]:$start:$MULTITHREADING_LEVEL}"
}

declare -A PID_BUF PID_SLOT
_running=0
_ok=0
_failed=0
_skipped=0
_SLOT_FREE=()
for (( _s = NJOBS - 1; _s >= 0; _s-- )); do _SLOT_FREE+=("$_s"); done

# Reap exactly one finished run, emit its output, and release its slot.
_reap_one() {
    local fp rc buf slot
    wait -n -p fp
    rc=$?
    # A trapped signal wakes wait -n with no child reaped; leave the map alone.
    if [ -z "${fp:-}" ]; then return 1; fi
    buf="${PID_BUF[$fp]}"
    slot="${PID_SLOT[$fp]}"
    unset "PID_BUF[$fp]" "PID_SLOT[$fp]"
    _dump_buffer "$buf"
    _SLOT_FREE+=("$slot")
    _running=$(( _running - 1 ))
    if [ "$rc" -eq 0 ]; then _ok=$(( _ok + 1 )); else _failed=$(( _failed + 1 )); fi
    return 0
}
'''

# Concurrent driver - emitted only when multithreading_level is configured.
PARALLEL_LOOP_TMPL = r'''
for idx in "${!INNER_ARGS[@]}"; do
    inner_args_str="${INNER_ARGS[$idx]}"
    line_no="${INNER_LINE_NOS[$idx]}"
    line_hash=$(printf '%s' "$EXEC_SIG${outer_args_str:+ $outer_args_str} $inner_args_str" | md5sum | cut -d ' ' -f 1)
    indicator="@@INDICATOR_PREFIX@@L${line_no}"
    CHECKPOINT_FILE="$CHECKPOINT_DIR/${indicator}_${line_hash}.done"

    # Runtime guard: protects a re-submitted or SLURM-requeued script. Checked
    # before dispatch so an already-completed point never occupies a slot.
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "[CHECKPOINT] Skipping $indicator - already completed."
        echo "[CHECKPOINT] Hash: ${line_hash}"
        _skipped=$(( _skipped + 1 ))
        continue
    fi

    # Degenerate case: stream straight to the log, exactly as the sequential path.
    if [ "$NJOBS" -le 1 ]; then
        run_inner_task "$line_no" "$inner_args_str" "$outer_args_str" "" "$CHECKPOINT_FILE"
        if [ $? -eq 0 ]; then _ok=$(( _ok + 1 )); else _failed=$(( _failed + 1 )); fi
        continue
    fi

    while [ "$_running" -ge "$NJOBS" ]; do
        _reap_one || break
    done

    _slot="${_SLOT_FREE[-1]}"
    unset '_SLOT_FREE[-1]'
    _buf="$LOG_BUF_DIR/${indicator}.log"
    : > "$_buf"
    run_inner_task "$line_no" "$inner_args_str" "$outer_args_str" "$(_slot_cpus "$_slot")" "$CHECKPOINT_FILE" >> "$_buf" 2>&1 &
    PID_BUF[$!]="$_buf"
    PID_SLOT[$!]="$_slot"
    _running=$(( _running + 1 ))
done
'''

# Drain for the concurrent driver. Deliberately outside the outer-combination
# wrapper: keeping one slot pool across boundaries lets the last runs of one
# combination overlap the first runs of the next instead of idling cores.
PARALLEL_TAIL_TMPL = r'''
while [ "$_running" -gt 0 ]; do
    _reap_one || break
done

echo "[SUMMARY] completed=$_ok skipped=$_skipped failed=$_failed"
wait
'''


def build_inner_driver(
    ctx: Dict[str, Any],
    config: AppConfig,
    indicator_prefix: str = "",
    wrap_outer_loop: bool = False,
) -> str:
    """Builds the shared run_inner_task function plus the loop that drives it.

    Emits the sequential driver unless execution.multithreading_level is set, in
    which case the concurrency runtime and dispatcher are emitted instead. Both
    drivers call the same run_inner_task, so the checkpoint/exec/timing logic
    exists in exactly one place and is identical across modes.

    With wrap_outer_loop the inner-loop body is nested inside a loop over
    MY_OUTER_IDS, so one array element can process several outer combinations.
    The drain stays outside that wrapper, which keeps the concurrency slot pool
    saturated across combination boundaries.
    """
    exec_cfg = config.execution
    level = exec_cfg.multithreading_level

    exec_cmd = build_exec_command(
        exec_cfg,
        ctx["flags_str"],
        ctx["fixed_args_str"],
        "$outer_args_str $inner_args_str",
    )

    concurrency_info = ""
    if level is not None:
        # Wall-clock duration inflates under contention, so record the conditions
        # it was measured under alongside it.
        concurrency_info = (
            '        printf "Concurrency: %d\\n" "$NJOBS"\n'
            '        printf "Threads: %d\\n" "$MULTITHREADING_LEVEL"'
        )

    task_fn = (
        RUN_INNER_TASK_TMPL
        .replace("@@INDICATOR_PREFIX@@", indicator_prefix)
        .replace("@@EXP_ARGS_BLOCK@@", ctx["exp_args_block"])
        .replace("@@FIXED_ARGS@@", ctx["fixed_args_str"])
        .replace("@@CONCURRENCY_INFO@@", concurrency_info)
        .replace("@@EXEC_CMD@@", exec_cmd)
    )

    def wrap(body: str) -> str:
        """Nests one inner-loop body inside the outer-combination loop."""
        if not wrap_outer_loop:
            return body
        return OUTER_WRAPPER_HEAD_TMPL + indent(body, "    ") + OUTER_WRAPPER_TAIL_TMPL

    if level is None:
        body = SEQUENTIAL_LOOP_TMPL.replace("@@INDICATOR_PREFIX@@", indicator_prefix)
        return task_fn + wrap(body) + SEQUENTIAL_TAIL_TMPL

    runtime = (
        PARALLEL_RUNTIME_TMPL
        .replace("@@MULTITHREADING_LEVEL@@", str(level))
        .replace("@@MAX_PARALLEL@@", str(compute_max_parallel(config)))
    )
    body = PARALLEL_LOOP_TMPL.replace("@@INDICATOR_PREFIX@@", indicator_prefix)
    return task_fn + runtime + wrap(body) + PARALLEL_TAIL_TMPL


def group_outer_tasks(
    pending_ids: List[int], num_array_jobs: Optional[int]
) -> List[Tuple[int, List[int]]]:
    """Maps pending outer combinations onto SLURM array elements.

    Returns (array_task_id, [outer combination indices]) pairs.

    Without num_array_jobs each pending combination becomes its own array element
    keyed by its *original* index, which reproduces the historical sparse
    '--array=0-2,5,8' behaviour exactly.

    With num_array_jobs the pending list - not the index space - is split into
    that many near-equal groups, so a resumed run still fills every element with
    real work. Group sizes differ by at most one, which also keeps the shared
    '--time' over-request to at most one combination's worth.
    """
    if num_array_jobs is None:
        return [(outer_id, [outer_id]) for outer_id in pending_ids]

    # Clamp: more elements than pending work would submit empty array tasks.
    groups = min(num_array_jobs, len(pending_ids))
    if groups <= 0:
        return []

    base, remainder = divmod(len(pending_ids), groups)
    assignments: List[Tuple[int, List[int]]] = []
    cursor = 0
    for task_id in range(groups):
        size = base + 1 if task_id < remainder else base
        assignments.append((task_id, pending_ids[cursor:cursor + size]))
        cursor += size
    return assignments


def compute_max_parallel(config: AppConfig) -> int:
    """Explicit ceiling on concurrent runs, or an effectively unlimited default.

    The CPU-derived limit is applied at runtime from SLURM_CPUS_PER_TASK; this is
    only the part that can be known at generation time.
    """
    cap = config.execution.max_parallel_tasks
    return cap if cap is not None else 9999


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


def build_single_mode(ctx: Dict[str, Any], config: AppConfig) -> Optional[str]:
    """Mode 0: Non-looped execution generator with Python pre-checkpoint check."""
    if is_single_mode_completed(ctx["checkpoint_dir"], ctx["exec_sig_str"]):
        print("[PRE-CHECK] Single run configuration already completed. Skipping SLURM generation.")
        return None

    exec_cmd = build_exec_command(
        ctx["exec_cfg"], ctx["flags_str"], ctx["fixed_args_str"], ""
    )
    return f"""{ctx['common_header']}

# Navigate to executable directory to preserve repo/git context
cd "{ctx['exec_dir']}" || exit 1

indicator="SINGLE"
EXEC_SIG="{ctx['exec_sig_str']}"
exec_hash=$(printf '%s' "$EXEC_SIG" | md5sum | cut -d ' ' -f 1)
CHECKPOINT_FILE="$CHECKPOINT_DIR/${{indicator}}_${{exec_hash}}.done"

# Runtime guard: protects a re-submitted or SLURM-requeued script
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

def build_inner_loop_mode(ctx: Dict[str, Any], config: AppConfig) -> Optional[str]:
    """Mode 1: Inner-loop-only generator pre-filtered in Python."""
    evaluated_inner = _evaluate_inner_loop(config.inner_loop)
    pending_tasks = filter_inner_loop_tasks(
        ctx["checkpoint_dir"], ctx["exec_sig_str"], evaluated_inner
    )

    if not pending_tasks:
        print("[PRE-CHECK] All inner loop tasks completed. Skipping SLURM generation.")
        return None

    bash_inner_args, bash_inner_line_nos = [], []
    for line_no, item_dict in pending_tasks:
        args_str = format_cli_args(item_dict)
        bash_inner_args.append(f'    "{args_str}"')
        bash_inner_line_nos.append(f"    {line_no}")

    args_array_block = "\n".join(bash_inner_args)
    line_nos_array_block = "\n".join(bash_inner_line_nos)
    arg_names = list(evaluated_inner[0].keys()) if evaluated_inner else []
    arg_names_display = ", ".join(arg_names)

    inner_driver = build_inner_driver(ctx, config, indicator_prefix="")

    return f"""{ctx['common_header']}

# Navigate to executable directory to preserve repo/git context
cd "{ctx['exec_dir']}" || exit 1

EXEC_SIG="{ctx['exec_sig_str']}"

# No outer loop in this mode; the shared inner driver expects the variable.
outer_args_str=""

INNER_ARGS=(
{args_array_block}
)

# Original 1-based indices from the full inner loop, preserved through
# pre-filtering so checkpoint names and log indicators stay stable across runs
INNER_LINE_NOS=(
{line_nos_array_block}
)

echo "======================= SLURM RUN LEGEND ======================="
echo "Mode: Inner Loop Only"
echo "Working Directory: {ctx['exec_dir']}"
echo "Exec Path: {ctx['exec_path']}"
echo "Fixed Args: {ctx['fixed_args_str']}"
echo "Inner Args: {arg_names_display}"
echo "Pending Tasks: {len(pending_tasks)} / {len(evaluated_inner)}"
echo "================================================================"
{inner_driver}"""

def _evaluate_outer_block(block: OuterLoopBlock) -> List[Dict[str, Any]]:
    """Evaluates a typed outer loop block into argument dictionaries."""
    if isinstance(block, ExplicitLoop):
        return [{block.arg_name: val} for val in block.values]

    elif isinstance(block, RangeLoop):
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
) -> Tuple[Optional[str], int]:
    """Mode 2: Outer + Inner loop SLURM Job Array using original explicit indices."""
    checkpoint_dir = ctx["checkpoint_dir"]
    inner_cfg = config.inner_loop
    outer_cfg = config.outer_loops

    # 1. Generate all original outer combinations
    evaluated_blocks = [_evaluate_outer_block(block) for block in outer_cfg]
    outer_combinations = []
    for combo_tuple in itertools.product(*evaluated_blocks):
        merged_combo = {}
        for d in combo_tuple:
            merged_combo.update(d)
        outer_combinations.append(merged_combo)

    evaluated_inner = _evaluate_inner_loop(inner_cfg)

    # 2. Identify incomplete task IDs based on original zero-indexed position
    pending_task_ids = get_pending_array_task_ids(
        checkpoint_dir, ctx["exec_sig_str"], outer_combinations, evaluated_inner
    )

    if not pending_task_ids:
        print("[PRE-CHECK] All job array tasks are complete. Skipping SLURM generation.")
        return None, 0

    # 3. Map pending outer combinations onto array elements. Without
    #    num_array_jobs this is one element per combination, keyed by the
    #    original index, so the array range stays sparse exactly as before.
    assignments = group_outer_tasks(pending_task_ids, config.slurm.num_array_jobs)
    packing = config.slurm.num_array_jobs is not None
    if packing:
        largest = max(len(members) for _, members in assignments)
        print(
            f"[INFO] Packing {len(pending_task_ids)} outer combination(s) into "
            f"{len(assignments)} array element(s); up to {largest} per element."
        )
        print(
            "[INFO] '--time' applies to every element, so it must cover the "
            f"largest ({largest} combination(s) run back to back)."
        )

    array_range_str = format_slurm_array_range(
        [task_id for task_id, _ in assignments], max_concurrent
    )

    # Strip existing #SBATCH --array directives and insert the calculated range after #!/bin/bash
    lines = [
        line for line in ctx["common_header"].splitlines() 
        if not line.strip().startswith("#SBATCH --array")
    ]
    lines.insert(1, f"#SBATCH --array={array_range_str}")
    header_with_array = "\n".join(lines)

    # 4. Build outer arrays containing ALL original combinations so index lookup aligns
    bash_outer_args, bash_outer_descs = [], []
    for combo_dict in outer_combinations:
        outer_args_str = format_cli_args(combo_dict)
        outer_desc = ", ".join(f"{strip_hyphens(k)}={v}" for k, v in combo_dict.items())
        bash_outer_args.append(f'    "{outer_args_str}"')
        bash_outer_descs.append(f'    "{outer_desc}"')

    outer_args_array_block = "\n".join(bash_outer_args)
    outer_descs_array_block = "\n".join(bash_outer_descs)

    # 5. Build inner parameter array
    bash_inner_args, all_inner_arg_names = [], set()
    for item in evaluated_inner:
        bash_inner_args.append(f'    "{format_cli_args(item)}"')
        all_inner_arg_names.update(item.keys())

    inner_args_array_block = "\n".join(bash_inner_args)
    arg_names_display = ", ".join(sorted(all_inner_arg_names)) if all_inner_arg_names else ""

    # The inner loop is not pre-filtered in this mode (outer tasks are), so the
    # driver's runtime checkpoint check is what skips completed points here.
    inner_line_nos_block = "\n".join(f"    {i}" for i in range(1, len(evaluated_inner) + 1))

    # Which outer combinations each array element owns, emitted as data rather
    # than recomputed with bash arithmetic: on a resume the pending indices are
    # sparse, so there is no dense index space to partition arithmetically.
    task_members_block = "\n".join(
        f'    [{task_id}]="{" ".join(str(m) for m in members)}"'
        for task_id, members in assignments
    )

    # The indicator keys on the outer combination index, never on the array task
    # id. With packing those differ, and keying on the task id would give two
    # different combinations the same indicator - and hence the same $SAVE_DIR
    # output directory. Unpacked they are equal, so existing checkpoints and
    # result directories keep their historical names.
    inner_driver = build_inner_driver(
        ctx, config, indicator_prefix="A${o_idx}_", wrap_outer_loop=True
    )

    script_content = f"""{header_with_array}

# Navigate to executable directory to preserve repo/git context
cd "{ctx['exec_dir']}" || exit 1

EXEC_SIG="{ctx['exec_sig_str']}"

# Complete outer loop array indexed by original task IDs
OUTER_ARGS=(
{outer_args_array_block}
)

OUTER_DESCS=(
{outer_descs_array_block}
)

INNER_ARGS=(
{inner_args_array_block}
)

# 1-based inner line numbers, matching the indicators used for checkpoint names
INNER_LINE_NOS=(
{inner_line_nos_block}
)

# Outer combinations owned by each array element (indices into OUTER_ARGS)
TASK_MEMBERS=(
{task_members_block}
)

TASK_ID=${{SLURM_ARRAY_TASK_ID:-0}}
read -ra MY_OUTER_IDS <<< "${{TASK_MEMBERS[$TASK_ID]}}"

echo "======================= SLURM ARRAY RUN LEGEND ======================="
echo "Array Task ID: $TASK_ID"
echo "Outer Combinations: ${{#MY_OUTER_IDS[@]}} (indices: ${{MY_OUTER_IDS[*]}})"
echo "Working Directory: {ctx['exec_dir']}"
echo "Exec Path: {ctx['exec_path']}"
echo "Fixed Args: {ctx['fixed_args_str']}"
echo "Inner Args: {arg_names_display}"
echo "Executing Subtasks: {len(pending_task_ids)} / {len(outer_combinations)} Outer Runs"
echo "======================================================================"
{inner_driver}"""
    return script_content, len(assignments)