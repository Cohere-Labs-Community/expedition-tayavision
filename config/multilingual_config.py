from dataclasses import dataclass, field

from config.training_config import InstructConfig


@dataclass
class MultilingualInstructConfig(InstructConfig):
    """Config for multilingual multimodal instruction tuning.

    Extends InstructConfig with multilingual mixing parameters based on
    Pangea [1] and mT5 [14] findings:
    - 40% English / 60% multilingual is optimal (Pangea ablation)
    - Temperature T=5 for strong low-resource upsampling (mT5)
    """

    # Total number of samples after mixing
    total_samples: int = 500_000

    # Fraction of total allocated to English (Pangea optimal: 0.4)
    english_ratio: float = 0.4

    # Temperature for non-English language sampling
    # T=1: proportional, T=5: strong upsampling (mT5), T→∞: uniform
    temperature: float = 5.0

    # Allow upsampling low-resource languages with replacement
    # When True, languages can be sampled beyond their available count
    # up to max_upsample_factor × available. Enables more uniform distribution.
    allow_upsampling: bool = True
    max_upsample_factor: float = 5.0

    # Path to instruct checkpoint (.pt) to initialise projector + LoRA from
    instruct_checkpoint: str = ""

    # HuggingFace cache directory for downloaded datasets
    hf_cache_dir: str = ""

    # NCCL timeout in minutes for DDP collectives (default 30 min)
    # Multilingual training has variable batch processing times across ranks
    nccl_timeout_minutes: int = 30

    # Dataset sources to mix — list of {name, max_samples, languages, data_dir}
    # Configured in YAML; defaults here for reference.
    # NOTE: PALO (MBZUAI/palo_multilingual_dataset) has NO language field —
    # all rows default to "en". Use Pangea instead for multilingual mixing.
    multilingual_sources: list = field(default_factory=lambda: [
        {
            "name": "pangea_ins",
            "max_samples": 300_000,
            "data_dir": "",
            "languages": [
                "en", "zh", "hi", "ar", "es", "fr", "ru", "ja", "ko",
                "id", "vi", "tr", "pt", "sw", "bn", "de", "it", "nl",
                "pl", "th", "ta", "te", "ur",
            ],
        },
        {
            "name": "aya_text",
            "max_samples": 100_000,
            "languages": [],  # empty = all languages
        },
    ])
