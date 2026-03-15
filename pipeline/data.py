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
        image = Image.open(image_path)

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

class InstructDataset(torch.utils.data.Dataset):
    """Dataset for instruction-finetuning with LLaVA-Instruct-150K.

    Multi-turn conversations with images, formatted using the chat_template
    from the instruction-tuned backbone (tiny-aya-global). Labels are masked
    so loss is computed only on assistant responses.
    """

    ROLE_MAP = {"human": "user", "gpt": "assistant"}

    def __init__(
        self,
        config: TinyAyaVisionConfig,
        data_dir: str = "data/llava-instruct",
        max_seq_len: int = 2048,
    ):
        self.data_dir = Path(data_dir)
        json_path = self.data_dir / "llava_instruct_150k.json"
        print(f"Loading dataset from {json_path}...")
        with open(json_path, "r") as f:
            self.dataset = json.load(f)
        # Keep only examples that have an image
        self.dataset = [x for x in self.dataset if "image" in x]
        print(f"Loaded {len(self.dataset)} examples")

        self.processor = TinyAyaVisionProcessor(config=config)
        self.max_seq_len = max_seq_len

        # Cache special token IDs for label masking
        self._chatbot_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|CHATBOT_TOKEN|>"
        )
        self._end_turn_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|END_OF_TURN_TOKEN|>"
        )

    def _to_chat_messages(self, conversations):
        """Convert LLaVA conversation format to chat template messages.

        The first user turn typically contains ``<image>\n`` which is
        converted to structured multimodal content
        ``[{"type": "image"}, {"type": "text", ...}]``.
        """
        messages = []
        for turn in conversations:
            role = self.ROLE_MAP[turn["from"]]
            value = turn["value"]

            if role == "user" and "<image>" in value:
                text = (
                    value.replace("<image>\n", "")
                    .replace("\n<image>", "")
                    .replace("<image>", "")
                    .strip()
                )
                content = [{"type": "image"}, {"type": "text", "text": text}]
            else:
                content = value

            messages.append({"role": role, "content": content})
        return messages

    def _build_labels(self, input_ids):
        """Mask labels so loss is computed only on assistant response tokens.

        Scans for ``<|CHATBOT_TOKEN|>`` (response start) and
        ``<|END_OF_TURN_TOKEN|>`` (response end) to identify the assistant
        spans.  The chatbot marker itself is masked; the end-of-turn token
        is included so the model learns to emit it.
        """
        labels = torch.full_like(input_ids, -100)
        in_response = False
        for i in range(len(input_ids)):
            tok = input_ids[i].item()
            if tok == self._chatbot_token_id:
                in_response = True
                continue
            if in_response and tok == self._end_turn_token_id:
                labels[i] = input_ids[i]
                in_response = False
                continue
            if in_response:
                labels[i] = input_ids[i]
        return labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_path = self.data_dir / "coco" / "train2017" / item["image"]
        image = Image.open(image_path).convert("RGB")

        messages = self._to_chat_messages(item["conversations"])

        # Full conversation formatted via chat_template (no generation prompt)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        processed = self.processor(
            text=full_text,
            images=image,
            truncation=True,
            max_length=self.max_seq_len,
        )

        input_ids = processed["input_ids"].squeeze(0)
        attention_mask = processed["attention_mask"].squeeze(0)
        pixel_values = processed["pixel_values"].squeeze(0)
        labels = self._build_labels(input_ids)

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

    