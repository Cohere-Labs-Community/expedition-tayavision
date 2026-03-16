import io
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from config.model_config import TinyAyaVisionConfig
from models.tiny_aya_vision import TinyAyaVisionForConditionalGeneration
from src.processing import TinyAyaVisionProcessor

HF_REPO = "bkal01/tayavision-alignment"

_IMAGENETTE_SYNSETS = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def load_imagenette_images(num_per_class: int = 1) -> list[tuple[str, Image.Image]]:
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    print("Downloading imagenette2-320 from FastAI...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tgz_path = Path(tmpdir) / "imagenette2-320.tgz"
        urllib.request.urlretrieve(url, tgz_path)
        result = []
        counts: dict[str, int] = {}
        with tarfile.open(tgz_path) as tf:
            for member in tf.getmembers():
                if not member.isfile() or not member.name.endswith(".JPEG"):
                    continue
                synset = member.name.split("/")[-2]
                if synset not in _IMAGENETTE_SYNSETS or counts.get(synset, 0) >= num_per_class:
                    continue
                f = tf.extractfile(member)
                if f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                    result.append((_IMAGENETTE_SYNSETS[synset], img))
                    counts[synset] = counts.get(synset, 0) + 1
                if all(counts.get(s, 0) >= num_per_class for s in _IMAGENETTE_SYNSETS):
                    break
        return result


def load_model(model_config: TinyAyaVisionConfig, device: torch.device):
    processor = TinyAyaVisionProcessor(config=model_config)
    model = TinyAyaVisionForConditionalGeneration(config=model_config)
    model.setup_tokenizer(processor.tokenizer)
    model.to(device)

    connector_path = hf_hub_download(HF_REPO, "connector.pt")
    state_dict = torch.load(connector_path, map_location=device)
    model.multi_modal_projector.load_state_dict(state_dict)
    model.eval()
    return model, processor
