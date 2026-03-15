from dataclasses import dataclass

@dataclass
class AlignmentConfig:
    """
    Config for aligning vision encoder w/LLM backbone via a learned connector.
    """
    dataset_name: str = "liuhaotian/LLaVA-Pretrain"
    data_dir: str = "/data/llava-pretrain"
    max_seq_len: int = 2048

    num_epochs: int = 1
    batch_size: int = 8
    grad_acc_steps: int = 32
    warmup_ratio: float = 0.03
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    num_workers: int = 4

    torch_dtype: str = "bfloat16"

    models_dir: str = "/models"
    save_steps: int = 500
    logging_steps: int = 10
    seed: int = 42


@dataclass
class InstructConfig(AlignmentConfig):
    """Config for instruction-finetuning with LLaVA-Instruct-150K."""

    dataset_name: str = "liuhaotian/LLaVA-Instruct-150K"
    data_dir: str = "/data/llava-instruct"

    models_dir: str = "/models"

    batch_size: int = 128  # global batch size across all GPUs(per-gpu batch size * world_size)
    grad_acc_steps: int = 32
    learning_rate: float = 2e-5

    # Path to alignment checkpoint (.pt) to initialise the projector from
    alignment_checkpoint: str = ""