from dataclasses import asdict, dataclass, field
from typing import Any, Literal
from TorchDist.model.model_args import ModelArgs

@dataclass
class Job:
    dump_folder:str = "./TorchDist/outputs"

@dataclass
class Profiling:
    enable_profiling:bool = False
    save_traces_floder:str = "profile_traces"
    profiler_freq:int = 10 # profile_freq: how frequently profiling is triggered by the training code.
    profiler_active:int=1 # profiler_active: number of steps for which profiling actively records.
    profiler_warmup:int=3 # profiler_warmup: number of warmup steps before collecting active profiling data.
    enable_memory_snapshot:bool = False # enable_memory_snapshot: enable memory snapshots for investigating memory usage.


@dataclass
class Metrics:
    log_freq: int = 10
    save_folder: str = "metrics"
    save_for_all_ranks: bool = False


@dataclass
class Model:
    hf_assets_path: str = ""
    args: ModelArgs = field(default_factory=ModelArgs)


@dataclass
class Optimizer:
    name: str = "AdamW"
    lr: float = 8e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    implementation: Literal["for-loop", "foreach", "fused"] = "fused"

@dataclass
class LRScheduler:
    warmup_steps:int=200
    decay_ratio:float | None = None
    decay_type: Literal["linear", "cosine", "sqrt"] = "linear"
    min_lr_factor:float = 0.0

@dataclass
class Training:
    dataset:str = "fineweb"
    dataset_path : str | None = None
    local_batch_size : int = 8
    global_batch_size : int = -1
    seq_len : int = 2048
    max_norm : float | int = 1.0
    steps : int = 10000
    dtype : Literal["bfloat16", "float32"] = "float32"
    mixed_precision : Literal["bfloat16", "float32"] = "bfloat16"
    gc_freq : int = 50
    seed: int | None = None
    deterministic : bool = False

@dataclass
class Parallelism:
    data_parallel_replicate_degree: int = 1
    enable_compiled_autograd: bool = False
    data_parallel_shard_degree: int = -1
    fsdp_reshard_after_forward: Literal["default", "always", "never"] = "default"
    tensor_parallel_degree: int = 1
    disable_loss_parallel: bool = False
    pipeline_parallel_degree: int = 1
    module_fqns_per_model_part: list[list[str]] | None = None
    pipeline_parallel_first_stage_less_layers: int = 1
    pipeline_parallel_last_stage_less_layers: int = 1
    pipeline_parallel_layers_per_stage: int | None = None
    pipeline_parallel_schedule: str = "1F1B"
    pipeline_parallel_microbatch_size: int = 1
    context_parallel_degree: int = 1
    context_parallel_rotate_method: Literal["allgather", "alltoall"] = "allgather"
    context_parallel_load_balance: bool = True
    expert_parallel_degree: int = 1
    expert_tensor_parallel_degree: int = 1

@dataclass
class Checkpoint:
    enable: bool = False
    folder: str = "checkpoint"
    interval: int = 500
    initial_load_path: str | None = None
    async_mode: Literal["disabled", "async"] = "disabled"
    keep_latest_k: int = 10
    load_step: int = -1
    enable_first_step_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.keep_latest_k == 1:
            raise ValueError(
                "The last checkpoint may still be in the process of being written, so keep_latest_k must be at least 2."
            )


@dataclass
class Checkpoint:
    enable: bool = False
    folder: str = "checkpoint"
    interval: int = 500
    initial_load_path: str | None = None
    async_mode: Literal["disabled", "async"] = "disabled"
    keep_latest_k: int = 10
    load_step: int = -1
    enable_first_step_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.keep_latest_k == 1:
            raise ValueError(
                "The last checkpoint may still be in the process of being written, so keep_latest_k must be at least 2."
            )


@dataclass
class ActivationCheckpoint:
    mode: Literal["selective", "full", "none"] = "selective"
    selective_ac_option: str = "2"
    per_op_sac_force_recompute_mm_shapes_by_fqns: list[str] = field(
        default_factory=lambda: ["moe.router.gate"]
    )




@dataclass
class Compile:
    enable: bool = False
    components: list[Literal["model", "loss"]] = field(
        default_factory=lambda: ["model", "loss"]
    )
    backend: str = "inductor"


@dataclass
class Comm:
    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 100



@dataclass
class JobConfig:
    """
    Default container for training configuration.
    """

    job: Job = field(default_factory=Job)
    profiling: Profiling = field(default_factory=Profiling)
    metrics: Metrics = field(default_factory=Metrics)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    comm: Comm = field(default_factory=Comm)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)