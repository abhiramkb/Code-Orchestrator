import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple, Literal, Optional, Union
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator, model_validator

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
        


# --- 1. Outer Loop Polymorphic Models ---

class ExplicitOuterLoop(BaseModel):
    type: Literal["explicit"] = "explicit"
    arg_name: str
    values: List[Any]


class RangeOuterLoop(BaseModel):
    type: Literal["range"]
    arg_name: str
    start: float
    stop: float
    step: float = 1.0


class TabularArgSpec(BaseModel):
    arg_name: str
    column: int
    template: Optional[str] = None
    transform: Optional[str] = None  # e.g., "10^x", "10**x", "log10", "exp"


class TabularOuterLoop(BaseModel):
    type: Literal["tabular_file"]
    file_path: Path
    args: List[TabularArgSpec]
    delimiter: Optional[str] = None
    comment_prefix: str = "#"
    skip_blank_lines: bool = True

    @field_validator("file_path")
    @classmethod
    def check_file_exists(cls, v: Path, info: ValidationInfo) -> Path:
        # Reads 'check_files' from context, defaulting to True if not provided
        check_files = info.context.get("check_files", True) if info.context else True        
        if check_files and not v.is_file():
            raise ValueError(f"Tabular file does not exist: '{v}'")
        return v


# Tagged Union discriminator routes validation based on the 'type' field
OuterLoopBlock = Annotated[
    Union[ExplicitOuterLoop, RangeOuterLoop, TabularOuterLoop],
    Field(discriminator="type"),
]


# --- 2. Sub-Section Models ---

class ExecutionConfig(BaseModel):
    interpreter: Optional[str] = Field(default=None)
    flags: Optional[List[str]] = Field(default_factory=list)
    executable: str
    modules: List[str] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)


class SlurmConfig(BaseModel):
    job_name: str = Field(default="orchestrator", alias="job-name")
    partition: str
    nodes: int = Field(default=1, alias="nodes") # Default number of nodes is 1 if not specified
    time: str
    output: str = Field(default="slurm_%j.log", alias="output") # Output set in setup_experiment_directories function if experiment tracking info is provided.
    max_concurrent_tasks: Optional[int] = Field(default=None, alias="max_concurrent_tasks")

    model_config = {"extra": "allow", "populate_by_name": True}


class InnerLoopConfig(BaseModel):
    file_path: Optional[Path] = None
    delimiter: str = Field(default=" ")
    comment_prefix: str = "#"
    args: List[TabularArgSpec] = []
    arg_names: Optional[List[str]] = None # Simpler specification for arg names without column mapping

    @field_validator("file_path")
    @classmethod
    def check_inner_file(cls, v: Optional[Path], info: ValidationInfo) -> Optional[Path]:
        # Reads 'check_files' from context, defaulting to True if not provided
        check_files = info.context.get("check_files", True) if info.context else True
        
        if check_files and v is not None and not v.is_file():
            raise ValueError(f"Inner loop file does not exist: '{v}'")
        return v
    
    @model_validator(mode="after")
    def migrate_arg_names(self) -> "InnerLoopConfig":
        """Converts 'arg_names' list into structured 'args' column specifications."""
        if not self.args and self.arg_names:
            self.args = [
                TabularArgSpec(arg_name=name, column=idx)
                for idx, name in enumerate(self.arg_names)
            ]
        if not self.args:
            raise ValueError("InnerLoopConfig requires 'args' (or simpler 'arg_names').")
        return self

    @property
    def arg_name_list(self) -> List[str]:
        """Convenience property for legends and validation checks."""
        return [spec.arg_name for spec in self.args]


class ExperimentConfig(BaseModel):
    result_database_path: str
    experiment_name: str
    slrm_output_dir: str
    tracking_args: Optional[Dict[str, str]] = None


# --- 3. Root Application Schema ---

class AppConfig(BaseModel):
    loopQ: Optional[bool] = None  # Optional explicit flag from config JSON
    execution: ExecutionConfig
    slurm: SlurmConfig
    inner_loop: Optional[InnerLoopConfig] = None
    outer_loops: List[OuterLoopBlock] = []
    experiment: Optional[ExperimentConfig] = None
    args: Dict[str, Any] = {}

    @property
    def mode_info(self) -> Tuple[bool, int]:
        """
        Replaces determine_loop_q logic.
        Returns:
            (is_loop_active: bool, mode_index: int)
            - mode_index 0: Single run mode (no loops)
            - mode_index 1: Inner loop only
            - mode_index 2: Inner and outer loops
        """
        # 1. Explicit loopQ flag present
        if self.loopQ is not None:
            if not self.loopQ:
                return False, 0
            mode = 2 if self.outer_loops else 1
            return True, mode

        # 2. Implicit check based on presence of inner/outer blocks
        has_inner = self.inner_loop is not None
        has_outer = len(self.outer_loops) > 0

        if not has_inner and not has_outer:
            return False, 0

        mode = 2 if has_outer else 1
        return True, mode

    @property
    def mode_index(self) -> int:
        """Helper property to access mode_index (0, 1, or 2) directly."""
        return self.mode_info[1]

    @property
    def is_loop_mode(self) -> bool:
        """Helper property to check if loop mode is active."""
        return self.mode_info[0]

    @model_validator(mode="after")
    def validate_mode_integrity(self) -> "AppConfig":
        """Enforces rules based on determined mode_index."""
        is_loop, mode = self.mode_info

        if is_loop:
            if self.inner_loop is None:
                raise ValueError("Missing 'inner_loop' section (required when loop mode is active).")
        else:
            # Single run mode
            if self.inner_loop is not None:
                raise ValueError("'inner_loop' should not be present when loop mode is disabled.")
            
        return self


# --- 4. Replacement Validation Entrypoint ---

def validate_config(cfg: dict, config_path: str, dryrunQ: bool) -> AppConfig:
    """Validates structure, types, and file existence using Pydantic."""
    try:
        validated_cfg = AppConfig.model_validate(cfg, context={"check_files": not dryrunQ})
        return validated_cfg
    except ValidationError as e:
        print(f"\n[CONFIG ERROR] Validation failed for '{config_path}':", file=sys.stderr)
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"])
            print(f"  - [{loc}]: {err['msg']}", file=sys.stderr)
        sys.exit(1)