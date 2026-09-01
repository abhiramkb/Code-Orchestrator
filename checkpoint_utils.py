import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from utils import format_cli_args

def compute_md5(sig_str: str) -> str:
    """Calculates MD5 hash matching Bash `printf '%s' "$sig_str" | md5sum`."""
    return hashlib.md5(sig_str.encode("utf-8")).hexdigest()

def format_slurm_array_range(indices: List[int], max_concurrent: Optional[int] = None) -> str:
    """
    Converts a list of integers into a compressed SLURM array range string.
    Example: [0, 1, 2, 5, 7, 8, 9] -> "0-2,5,7-9"
    """
    if not indices:
        return ""
    
    indices = sorted(indices)
    ranges = []
    start = end = indices[0]

    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            ranges.append(f"{start}-{end}" if start != end else f"{start}")
            start = end = idx
    ranges.append(f"{start}-{end}" if start != end else f"{start}")

    range_str = ",".join(ranges)
    if max_concurrent:
        range_str += f"%{max_concurrent}"
    return range_str


def get_pending_array_task_ids(
    checkpoint_dir: Path,
    exec_sig_str: str,
    outer_combinations: List[Dict[str, Any]],
    evaluated_inner: List[Dict[str, Any]],
) -> List[int]:
    """Returns original task indices for outer combinations that have pending work."""
    pending_task_ids = []

    for task_id, combo_dict in enumerate(outer_combinations):
        outer_args_str = format_cli_args(combo_dict)
        combo_has_pending = False

        for line_no, inner_item in enumerate(evaluated_inner, start=1):
            inner_args_str = format_cli_args(inner_item)
            sig = f"{exec_sig_str} {outer_args_str} {inner_args_str}"
            combo_hash = compute_md5(sig)
            chk_file = checkpoint_dir / f"A{task_id}_L{line_no}_{combo_hash}.done"

            if not chk_file.is_file():
                combo_has_pending = True
                break

        if combo_has_pending:
            pending_task_ids.append(task_id)

    return pending_task_ids

def is_single_mode_completed(checkpoint_dir: Path, exec_sig_str: str) -> bool:
    """Checks if Single Run execution has already completed."""
    exec_hash = compute_md5(exec_sig_str)
    chk_file = checkpoint_dir / f"SINGLE_{exec_hash}.done"
    return chk_file.is_file()


def filter_inner_loop_tasks(
    checkpoint_dir: Path,
    exec_sig_str: str,
    evaluated_inner: List[Dict[str, Any]],
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Returns only incomplete inner loop items along with their original 1-based line index.
    """
    pending_tasks = []
    for line_no, item in enumerate(evaluated_inner, start=1):
        inner_args_str = format_cli_args(item)
        sig = f"{exec_sig_str} {inner_args_str}"
        line_hash = compute_md5(sig)
        chk_file = checkpoint_dir / f"L{line_no}_{line_hash}.done"

        if not chk_file.is_file():
            pending_tasks.append((line_no, item))

    return pending_tasks
