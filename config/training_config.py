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
    # None = run the full num_epochs (Modal behaviour, unchanged). An int caps
    # optimizer steps, which is what makes a bounded TPU smoke stoppable without
    # killing the tmux session.
    max_steps: int | None = None
    # None = pad each batch to its own longest item (CUDA). An int pins the
    # sequence length, which XLA requires to avoid recompiling per batch shape.
    fixed_seq_len: int | None = None
    warmup_ratio: float = 0.03
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    embed_align_reg: float = 0.0
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

    # Save merged HF-compatible model after training
    save_hf_model: bool = False

    # Optionally merge the VLM's LLM backbone with the original base LLM
    # using linear interpolation (LERP) after LoRA merge.
    merge_with_base_llm: bool = False
    base_llm_name: str = ""       # e.g. "CohereForAI/tiny-aya-global"
    merge_alpha: float = 0.7      # 0 = pure base LLM, 1 = pure VLM