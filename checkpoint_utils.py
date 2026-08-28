import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
from utils import format_cli_args

def compute_md5(sig_str: str) -> str:
    """Calculates MD5 hash matching Bash `printf '%s' "$sig_str" | md5sum`."""
    return hashlib.md5(sig_str.encode("utf-8")).hexdigest()


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


def filter_array_mode_tasks(
    checkpoint_dir: Path,
    exec_sig_str: str,
    outer_combinations: List[Dict[str, Any]],
    evaluated_inner: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Returns only outer combinations that have at least one uncompleted inner loop task.
    """
    pending_outer_combinations = []

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
                break  # Outer task needs to run

        if combo_has_pending:
            pending_outer_combinations.append(combo_dict)

    return pending_outer_combinations