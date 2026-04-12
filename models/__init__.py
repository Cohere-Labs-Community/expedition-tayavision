from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor

from config.model_config import TinyAyaVisionConfig
from src.processing import TinyAyaVisionProcessor
from .tiny_aya_vision import TinyAyaVisionForConditionalGeneration, TinyAyaVisionOutput

if TYPE_CHECKING:
    pass

AutoConfig.register("tiny_aya_vision", TinyAyaVisionConfig)
AutoModel.register(TinyAyaVisionConfig, TinyAyaVisionForConditionalGeneration)
AutoModelForCausalLM.register(TinyAyaVisionConfig, TinyAyaVisionForConditionalGeneration)
AutoProcessor.register(TinyAyaVisionConfig, TinyAyaVisionProcessor)


def save_for_inference(
    model: TinyAyaVisionForConditionalGeneration,
    processor: TinyAyaVisionProcessor,
    output_dir: str | Path,
) -> None:
    """Save the full model in HuggingFace-compatible format.

    After calling this, the model can be loaded with:
        import models  # registers Auto classes
        model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)

    Args:
        model: Trained TinyAyaVision model.
        processor: The processor (tokenizer + image processor are saved).
        output_dir: Directory to write config.json + model weights.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


__all__ = [
    "TinyAyaVisionConfig",
    "TinyAyaVisionForConditionalGeneration",
    "TinyAyaVisionOutput",
    "save_for_inference",
]
