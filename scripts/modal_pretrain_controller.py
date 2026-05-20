"""Modal entrypoint for Stage-1 SCTAC controller pre-training.

Trains only the script_controller MLP (everything else frozen) on a
language-balanced subsample of the same multilingual sources used by
the joint multilingual recipe. Single-GPU only — the controller is 387
parameters; DDP/torch.compile is pure overhead.

Prerequisites:
    1. Data already populated in the `multilingual-data` volume.
       (modal run scripts/modal_download_multilingual_data.py)
    2. A valid instruct checkpoint reachable at /models/<run_id>/<file>.pt.
    3. data/fertility/qwen3.json present (generated locally via
       scripts/compute_fertility_table.py).
    4. Modal secrets: huggingface, wandb.

Usage:
    modal run scripts/modal_pretrain_controller.py \\
        --instruct-checkpoint /models/<run_id>/checkpoint_<step>.pt

    # Smoke test
    modal run scripts/modal_pretrain_controller.py \\
        --instruct-checkpoint ... --debug-steps 200

    # Different controller variant
    modal run scripts/modal_pretrain_controller.py \\
        --instruct-checkpoint ... --controller b2
"""

import os

import modal

GPU_TYPE = os.environ.get("MODAL_GPU", "A100-40GB")

app = modal.App("tayavision-pretrain-controller")

data_volume = modal.Volume.from_name("multilingual-data")
models_volume = modal.Volume.from_name("tayavision-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .uv_pip_install(
        "torch==2.9.1",
        "torchvision",
        "transformers==4.56.2",
        "accelerate",
        "huggingface_hub",
        "tokenizers",
        "sentencepiece",
        "protobuf",
        "datasets>=2.16.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "orjson>=3.9.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "pyyaml",
        "peft",
        "wandb",
        "einops",
        "tqdm",
        "Pillow",
        "numpy",
    )
    .add_local_dir("config", remote_path="/root/project/config")
    .add_local_dir("src", remote_path="/root/project/src")
    .add_local_dir("pipeline", remote_path="/root/project/pipeline")
    .add_local_dir("models", remote_path="/root/project/models")
    .add_local_dir("data/fertility", remote_path="/root/project/data/fertility")
)


def _fix_source_data_dirs(multilingual_sources: list) -> None:
    """Mirror of fix_source_data_dirs in modal_train_multilingual.py."""
    source_to_path = {
        "pangea_ins": "/data/pangea-instruct",
        "palo": "/data/palo",
        "llava_instruct": "/data/llava-instruct",
        "mvl_sib": "/data/mvl-sib",
        "wit": "/data/wit",
        "bloom_lm": "/data/bloom-lm",
        "alm_bench": "/data/alm-bench",
        "bloom_captioning": "/data/bloom-captioning",
        "aya_text": "/data/aya-dataset",
    }
    for source in multilingual_sources:
        name = source.get("name", "")
        if name in source_to_path and not source.get("data_dir"):
            source["data_dir"] = source_to_path[name]


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": data_volume, "/models": models_volume},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3600 * 6,  # 6 hours; typical Stage 1 runs in 1-3 hours
)
def pretrain(
    instruct_checkpoint: str,
    backbone: str = "qwen3",
    controller: str = "b1",
    total_samples: int = 49_000,
    english_ratio: float = 0.015,
    temperature: float = 1.0e6,
    batch_size: int = 16,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 10,
    target_tokens: int = 98,
    rate_lambda: float = 0.1,
    debug_steps: int | None = None,
):
    """Run Stage-1 controller pre-training on a single GPU.

    Args:
        instruct_checkpoint: Path inside the models volume, e.g.
            /models/<run_id>/checkpoint_<step>.pt. Required.
        backbone: LLM backbone (qwen3 default).
        controller: Controller variant (b1, b2, b3). Stage 1 with `off` is rejected.
        total_samples: ~700 × N_langs. Default 49k.
        english_ratio: ≈ 1/N_langs for uniform sampling.
        temperature: 1e6 ≈ uniform across non-English languages.
        batch_size: 16 (per-GPU, single-GPU only).
        learning_rate: 1e-3, much higher than instruct LR (LoRA: 2e-5).
        num_epochs: 10 over the 49k mix.
        target_tokens: Scalar reference budget; per-sample target scales by tpw.
        rate_lambda: Rate-loss coefficient (matches controller yaml default).
        debug_steps: If set, smoke-test recipe (small batch, N steps).
    """
    import sys

    sys.path.insert(0, "/root/project")
    os.environ["HF_HOME"] = "/data/.hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/.hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/data/.hf_cache"

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from config.lora_config import LoraAdapterConfig
    from config.model_config import TinyAyaVisionConfig
    from config.multilingual_config import MultilingualInstructConfig
    from pipeline.pretrain_controller import main as pretrain_main

    if controller == "off":
        raise SystemExit("Stage 1 requires controller=b1 (or b2/b3). Got off.")

    print("=" * 60)
    print("Stage-1 SCTAC Controller Pre-training on Modal")
    print("=" * 60)
    print(f"GPU: {GPU_TYPE}")
    print(f"Backbone: {backbone}    Controller: {controller}")
    print(f"Total samples: {total_samples:,}   English ratio: {english_ratio:.3f}")
    print(f"Temperature: {temperature:.0e}    Batch: {batch_size}")
    print(f"LR: {learning_rate}    Epochs: {num_epochs}    target_tokens: {target_tokens}")
    print(f"Instruct ckpt: {instruct_checkpoint}")
    print(f"Debug steps: {debug_steps or 'off (full run)'}")
    print("=" * 60)

    overrides = [
        f"backbone={backbone}",
        f"controller={controller}",
        "training=controller_pretrain",
        "training.models_dir=/models/controller_pretrain_ckpts",
        "training.hf_cache_dir=/data/.hf_cache",
        f"training.total_samples={total_samples}",
        f"training.english_ratio={english_ratio}",
        f"training.temperature={temperature}",
        f"training.batch_size={batch_size}",
        f"training.learning_rate={learning_rate}",
        f"training.num_epochs={num_epochs}",
        f"training.instruct_checkpoint={instruct_checkpoint}",
        f"controller.target_tokens={target_tokens}",
        f"controller.rate_lambda={rate_lambda}",
    ]
    if debug_steps is not None:
        overrides.extend([
            "training.batch_size=4",
            "training.grad_acc_steps=1",
            "training.save_steps=20",
            "training.num_epochs=1",
        ])

    with initialize_config_dir(config_dir="/root/project/config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

        training_dict = OmegaConf.to_container(cfg.training, resolve=True)
        assert isinstance(training_dict, dict)
        lora_dict = training_dict.pop("lora", {})
        multilingual_sources = training_dict.pop("multilingual_sources", [])
        _fix_source_data_dirs(multilingual_sources)

        training_config = MultilingualInstructConfig(
            **training_dict, multilingual_sources=multilingual_sources,
        )

        backbone_name = (
            cfg.get("backbone", {}).get("backbone_type", "tiny_aya")
            if "backbone" in cfg
            else "tiny_aya"
        )
        model_config = TinyAyaVisionConfig.for_backbone(
            backbone=backbone_name,
            encoder=cfg.vision.vision_encoder_type,
        )
        for group_name in ("vision", "backbone"):
            if group_name in cfg:
                for k, v in OmegaConf.to_container(cfg[group_name], resolve=True).items():
                    if hasattr(model_config, k):
                        setattr(model_config, k, v)
        if model_config.backbone_type == "tiny_aya":
            model_config.llm_model_name = {
                "base": "CohereLabs/tiny-aya-base",
                "global": "CohereLabs/tiny-aya-global",
            }[cfg.llm]
        if "controller" in cfg:
            model_config.controller_config = OmegaConf.to_container(
                cfg.controller, resolve=True,
            )

        if "target_modules" not in lora_dict and model_config.lora_target_modules:
            lora_dict["target_modules"] = list(model_config.lora_target_modules)
        lora_config = LoraAdapterConfig(**lora_dict)

        pretrain_main(
            training_config=training_config,
            model_config=model_config,
            lora_config=lora_config,
        )


@app.local_entrypoint()
def main(
    instruct_checkpoint: str,
    backbone: str = "qwen3",
    controller: str = "b1",
    total_samples: int = 49_000,
    english_ratio: float = 0.015,
    temperature: float = 1.0e6,
    batch_size: int = 16,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 10,
    target_tokens: int = 98,
    rate_lambda: float = 0.1,
    debug_steps: int | None = None,
):
    pretrain.remote(
        instruct_checkpoint=instruct_checkpoint,
        backbone=backbone,
        controller=controller,
        total_samples=total_samples,
        english_ratio=english_ratio,
        temperature=temperature,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        target_tokens=target_tokens,
        rate_lambda=rate_lambda,
        debug_steps=debug_steps,
    )
