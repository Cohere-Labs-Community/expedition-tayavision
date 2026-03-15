import json
import torch

from pathlib import Path
from PIL import Image

from config.model_config import TinyAyaVisionConfig
from src.processing import TinyAyaVisionProcessor

class AlignmentDataset(torch.utils.data.Dataset):
    """
    Dataset for aligning vision encoder w/LLM backbone via a learned connector.
    LLaVA-Pretrain dataset with image-caption pairs.
    """
    def __init__(
        self,
        config: TinyAyaVisionConfig,
        dataset_name: str = "liuhaotian/LLaVA-Pretrain",
        data_dir: str = "data/llava-pretrain",
    ):
        self.data_dir = Path(data_dir)
        print(f"Loading dataset from {self.data_dir / 'blip_laion_cc_sbu_558k.json'}...")
        with open(self.data_dir / "blip_laion_cc_sbu_558k.json", "r") as f:
            self.dataset = json.load(f)
        print(f"Loaded {len(self.dataset)} examples")
        self.processor = TinyAyaVisionProcessor(
            config=config,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = self.data_dir / item["image"]
        image = Image.open(image_path).convert("RGB")

        prompt = item["conversations"][0]["value"]
        response = item["conversations"][1]["value"]

        processed = self.processor(
            images=image,
            text=prompt + response,
        )
        input_ids = processed["input_ids"].squeeze(0)
        attention_mask = processed["attention_mask"].squeeze(0)
        pixel_values = processed["pixel_values"].squeeze(0)

        processed_prompt = self.processor(
            text=prompt,
        )
        num_prompt_tokens = processed_prompt["input_ids"].shape[-1]
        labels = input_ids.clone()
        labels[:num_prompt_tokens] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }

def collate_fn(
    batch,
    pad_token_id: int,
):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    labels = torch.nn.utils.rnn.pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True,
        padding_value=-100,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }

    