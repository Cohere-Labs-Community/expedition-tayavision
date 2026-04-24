"""Multilingual multimodal dataset mixer for Phase 2 instruction tuning.

Implements the data mixing strategy from the multilingual training research:
- Temperature-based language sampling (mT5/BLOOM style)
- 40% English / 60% multilingual optimal ratio (Pangea finding)
- Cross-lingual transfer via language family grouping
- Multiple HuggingFace dataset source support

Datasets are loaded from HuggingFace Hub, language-tagged, and mixed into a
single interleaved stream with configurable sampling weights.

Dataset schemas (verified from HF / GitHub):
- PangeaIns: {id, image (path str), conversations [{from, value}], language (2-letter code)}
- PALO: {id, image (path str), conversations [{from, value}]} — NO language field
- Aya: {inputs, targets, language (full name), language_code (ISO 639-3)}
- LLaVA-Instruct: {id, image (path str), conversations [{from, value}]} — English only
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset as HFDataset
from huggingface_hub import hf_hub_download
from PIL import Image

from config.model_config import TinyAyaVisionConfig
from src.processing import TinyAyaVisionProcessor


# ---------------------------------------------------------------------------
# Language family mapping for cross-lingual transfer exploitation
# ---------------------------------------------------------------------------

LANGUAGE_FAMILIES = {
    "romance": ["es", "fr", "pt", "it", "ro", "ca", "gl"],
    "slavic": ["ru", "pl", "uk", "cs", "sk", "hr", "sr", "bg", "sl"],
    "indic": ["hi", "mr", "bn", "gu", "pa", "ne", "ur"],
    "dravidian": ["ta", "te"],
    "east_asian": ["zh", "ja", "ko"],
    "southeast_asian": ["vi", "th", "id", "ms", "jv", "lo", "km", "my", "tl"],
    "semitic_me": ["ar", "he", "fa", "tr"],
    "germanic": ["en", "de", "nl", "da", "sv", "no"],
    "celtic_baltic": ["cy", "ga", "lv", "lt", "et"],
    "finno_ugric": ["fi", "hu"],
    "bantu_african": ["sw", "sn", "xh", "zu", "yo", "ha", "ig", "am", "wo", "mg"],
    "other": ["el", "eu", "mt", "la"],
}

LANG_TO_FAMILY = {}
for family, langs in LANGUAGE_FAMILIES.items():
    for lang in langs:
        LANG_TO_FAMILY[lang] = family

# Anchor languages: invest most translation/data effort here,
# others benefit from cross-lingual transfer within the family.
ANCHOR_LANGUAGES = [
    "en", "zh", "hi", "ar", "es", "fr", "ru", "ja", "ko",
    "id", "vi", "tr", "pt", "sw", "bn", "de",
]


def compute_temperature_weights(
    lang_counts: dict[str, int],
    temperature: float = 5.0,
) -> dict[str, float]:
    """Compute per-language sampling probabilities using temperature scaling.

    p_i = n_i^(1/T) / sum_j(n_j^(1/T))

    T=1: proportional (high-resource dominates)
    T=5: strong upsampling of low-resource (mT5's choice)
    T→∞: uniform across all languages
    """
    langs = list(lang_counts.keys())
    counts = np.array([lang_counts[l] for l in langs], dtype=np.float64)

    # Avoid division by zero for empty languages
    counts = np.maximum(counts, 1.0)

    scaled = np.power(counts, 1.0 / temperature)
    probs = scaled / scaled.sum()

    return {lang: float(prob) for lang, prob in zip(langs, probs)}


def compute_sampling_indices(
    lang_counts: dict[str, int],
    total_samples: int,
    temperature: float = 5.0,
    english_ratio: float = 0.4,
    seed: int = 42,
    allow_upsampling: bool = True,
    max_upsample_factor: float = 5.0,
) -> dict[str, int]:
    """Compute how many samples to draw from each language.

    Enforces the English ratio constraint (default 40%) and distributes
    the remaining budget across non-English languages using temperature
    sampling.

    When allow_upsampling=True (default), low-resource languages can be
    sampled with replacement up to max_upsample_factor × their available
    data. This enables more uniform distribution across languages.
    When False, each language is capped at its available count.
    """
    rng = np.random.RandomState(seed)

    en_count = lang_counts.get("en", 0)
    non_en_counts = {l: c for l, c in lang_counts.items() if l != "en"}

    # Allocate English budget
    en_budget = int(total_samples * english_ratio)
    en_budget = min(en_budget, en_count)

    # Distribute remaining budget across non-English languages
    non_en_budget = total_samples - en_budget
    if non_en_counts:
        non_en_weights = compute_temperature_weights(non_en_counts, temperature)
        target_per_lang = {
            l: int(non_en_budget * w) for l, w in non_en_weights.items()
        }
        for l in target_per_lang:
            if allow_upsampling:
                # Allow upsampling up to max_upsample_factor × available data
                cap = int(non_en_counts[l] * max_upsample_factor)
                target_per_lang[l] = min(target_per_lang[l], cap)
            else:
                # Hard cap at available data
                target_per_lang[l] = min(target_per_lang[l], non_en_counts[l])
    else:
        target_per_lang = {}

    # Add English
    target_per_lang["en"] = en_budget

    return target_per_lang


# ---------------------------------------------------------------------------
# Dataset source definitions — each source knows how to extract
# (language, conversations, image) from its HF schema.
# ---------------------------------------------------------------------------

class DatasetSource:
    """Base class for a multilingual dataset source."""

    name: str = ""
    hf_dataset_id: str = ""
    hf_subset: str | None = None
    split: str = "train"
    languages: list[str] = []

    def __init__(self, data_dir: str | None = None):
        """
        Args:
            data_dir: Local directory for resolving image file paths.
                Required for datasets that store image paths (Pangea, PALO)
                rather than embedded PIL images.
        """
        self.data_dir = Path(data_dir) if data_dir else None

    def load(self, cache_dir: str | None = None, max_samples: int | None = None) -> HFDataset:
        """Load dataset from HuggingFace Hub.

        For standard HF datasets (Aya, LLaVA-Instruct), uses `load_dataset`.
        Subclasses that need custom loading (Pangea) override this method.
        """
        kwargs = {"cache_dir": cache_dir, "split": self.split}
        if self.hf_subset:
            kwargs["name"] = self.hf_subset
        ds = load_dataset(self.hf_dataset_id, **kwargs, trust_remote_code=True)
        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds

    def get_language(self, example: dict) -> str:
        raise NotImplementedError

    def to_conversations(self, example: dict) -> list[dict]:
        """Convert to unified format: [{"from": "human"/"gpt", "value": str}, ...]"""
        raise NotImplementedError

    @staticmethod
    def _parse_conversations(convs) -> list[dict]:
        """Deserialize conversations that may be a JSON string (from Parquet) or a list."""
        if isinstance(convs, str):
            return json.loads(convs)
        return convs

    def _resolve_image(self, img_field) -> Image.Image | None:
        """Resolve image from a HF dataset field.

        HF datasets may store images as:
        - PIL Image (when using datasets Image feature) → return directly
        - str path → open from data_dir / path
        - None → return None
        """
        if img_field is None:
            return None
        if isinstance(img_field, Image.Image):
            return img_field.convert("RGB")
        if isinstance(img_field, str) and self.data_dir is not None:
            img_path = self.data_dir / img_field
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
        return None

    def get_image(self, example: dict) -> Image.Image | None:
        return self._resolve_image(example.get("image"))


class PangeaInsSource(DatasetSource):
    """neulab/PangeaInstruct (6M, 39 languages).

    Verified schema: {id, image (path str), conversations [{from, value}], language (2-letter)}
    Images are file paths relative to a data_dir (requires separate download).
    The `language` field uses 2-letter ISO codes (e.g. "ko", "en", "zh").

    This is NOT a standard HF dataset — it's a file repo with PangeaIns.json.
    The load() method downloads the JSON via hf_hub_download and converts to
    an Arrow-backed HF Dataset.
    """
    name = "pangea_ins"
    hf_dataset_id = "neulab/PangeaInstruct"

    def load(self, cache_dir: str | None = None, max_samples: int | None = None) -> HFDataset:
        """Download PangeaIns.json and load as HF Dataset.

        Uses a Parquet cache for fast reloads. First load parses JSON with
        orjson (falling back to stdlib json) and writes Parquet; subsequent
        loads read the Parquet directly (~seconds vs 10-30 min for raw JSON).
        """
        local_dir = str(self.data_dir) if self.data_dir else cache_dir or "."

        json_path = Path(local_dir) / "PangeaIns.json"
        parquet_path = Path(local_dir) / "PangeaIns.parquet"

        if not json_path.exists():
            print(f"    Downloading PangeaIns.json from {self.hf_dataset_id}...")
            hf_hub_download(
                repo_id=self.hf_dataset_id,
                filename="PangeaIns.json",
                repo_type="dataset",
                local_dir=local_dir,
            )
            print(f"    Downloaded to {json_path}")

        # Fast path: load from cached Parquet
        if parquet_path.exists():
            print(f"    Loading PangeaIns from cached Parquet...")
            ds = HFDataset.from_parquet(str(parquet_path))
            if max_samples and len(ds) > max_samples:
                ds = ds.shuffle(seed=42).select(range(max_samples))
            return ds

        # First load: parse JSON → Arrow → Parquet cache
        print(f"    Parsing PangeaIns.json ({json_path.stat().st_size / 1e9:.1f} GB)...")
        try:
            import orjson
            with open(json_path, "rb") as f:
                records = orjson.loads(f.read())
            print(f"    Parsed {len(records):,} records with orjson")
        except ImportError:
            print(f"    orjson not available, falling back to stdlib json (slower)...")
            with open(json_path, "r") as f:
                records = json.load(f)
            print(f"    Parsed {len(records):,} records with json")

        # Convert conversations list to JSON strings for Arrow compatibility
        for r in records:
            if "conversations" in r and isinstance(r["conversations"], list):
                r["conversations"] = json.dumps(r["conversations"])

        print(f"    Converting to Arrow dataset...")
        df = pd.DataFrame(records)
        del records  # free ~12 GB
        # Coerce mixed-type columns to strings to avoid ArrowTypeError
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).replace("None", pd.NA).replace("nan", pd.NA)
        ds = HFDataset.from_pandas(df)
        del df

        # Cache as Parquet for fast future loads
        print(f"    Caching as Parquet at {parquet_path}...")
        ds.to_parquet(str(parquet_path))
        print(f"    Parquet cache saved ({parquet_path.stat().st_size / 1e9:.1f} GB)")

        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds

    def get_language(self, example: dict) -> str:
        return example.get("language", "en")

    def to_conversations(self, example: dict) -> list[dict]:
        return self._parse_conversations(example["conversations"])


class PaloSource(DatasetSource):
    """MBZUAI/palo_multilingual_dataset (2.1M, 10 languages).

    Verified schema: {id, image (path str), conversations [{from, value}]}
    WARNING: No `language` field. The dataset is 665K English (LLaVA mix665k)
    + 150K each translated to: zh, fr, es, ru, ja, ar, hi, bn, ur.
    Images are file paths relative to a data_dir (COCO, GQA, etc.).

    This is a single 13.3GB JSON file on HF, not a standard dataset.
    """
    name = "palo"
    hf_dataset_id = "MBZUAI/palo_multilingual_dataset"

    def load(self, cache_dir: str | None = None, max_samples: int | None = None) -> HFDataset:
        """Download palo_multilingual_dataset.json and load as HF Dataset.

        Uses a Parquet cache for fast reloads.
        """
        local_dir = str(self.data_dir) if self.data_dir else cache_dir or "."

        json_path = Path(local_dir) / "palo_multilingual_dataset.json"
        parquet_path = Path(local_dir) / "palo_multilingual_dataset.parquet"

        if not json_path.exists() and not parquet_path.exists():
            print(f"    Downloading palo_multilingual_dataset.json (~13GB)...")
            hf_hub_download(
                repo_id=self.hf_dataset_id,
                filename="palo_multilingual_dataset.json",
                repo_type="dataset",
                local_dir=local_dir,
            )
            print(f"    Downloaded to {json_path}")

        # Fast path: load from cached Parquet
        if parquet_path.exists():
            print(f"    Loading PALO from cached Parquet...")
            ds = HFDataset.from_parquet(str(parquet_path))
            if max_samples and len(ds) > max_samples:
                ds = ds.shuffle(seed=42).select(range(max_samples))
            return ds

        # First load: parse JSON → Arrow → Parquet cache
        print(f"    Parsing palo_multilingual_dataset.json ({json_path.stat().st_size / 1e9:.1f} GB)...")
        try:
            import orjson
            with open(json_path, "rb") as f:
                records = orjson.loads(f.read())
            print(f"    Parsed {len(records):,} records with orjson")
        except ImportError:
            print(f"    orjson not available, falling back to stdlib json (slower)...")
            with open(json_path, "r") as f:
                records = json.load(f)
            print(f"    Parsed {len(records):,} records with json")

        for r in records:
            if "conversations" in r and isinstance(r["conversations"], list):
                r["conversations"] = json.dumps(r["conversations"])

        print(f"    Converting to Arrow dataset...")
        df = pd.DataFrame(records)
        del records
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).replace("None", pd.NA).replace("nan", pd.NA)
        ds = HFDataset.from_pandas(df)
        del df

        print(f"    Caching as Parquet at {parquet_path}...")
        ds.to_parquet(str(parquet_path))
        print(f"    Parquet cache saved ({parquet_path.stat().st_size / 1e9:.1f} GB)")

        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds

    def get_language(self, example: dict) -> str:
        # PALO has no language field — all rows default to "en".
        # Callers should be aware of this limitation.
        return example.get("language", "en")

    def to_conversations(self, example: dict) -> list[dict]:
        return self._parse_conversations(example["conversations"])


class LLaVAInstructSource(DatasetSource):
    """English-only LLaVA-Instruct-150K loaded from HF Hub.

    Schema: {id, conversations [{from, value}], image (path str)}
    This is a JSON-only dataset on HF (no images embedded).
    Images require separate download of COCO train2017 etc.
    """
    name = "llava_instruct"
    hf_dataset_id = "liuhaotian/LLaVA-Instruct-150K"

    def load(self, cache_dir: str | None = None, max_samples: int | None = None) -> HFDataset:
        """Download llava_instruct_150k.json and load as HF Dataset.

        Uses a Parquet cache for fast reloads.
        """
        local_dir = str(self.data_dir) if self.data_dir else cache_dir or "."

        json_path = Path(local_dir) / "llava_instruct_150k.json"
        parquet_path = Path(local_dir) / "llava_instruct_150k.parquet"

        if not json_path.exists() and not parquet_path.exists():
            print(f"    Downloading llava_instruct_150k.json...")
            hf_hub_download(
                repo_id=self.hf_dataset_id,
                filename="llava_instruct_150k.json",
                repo_type="dataset",
                local_dir=local_dir,
            )
            print(f"    Downloaded to {json_path}")

        # Fast path: load from cached Parquet
        if parquet_path.exists():
            print(f"    Loading LLaVA-Instruct from cached Parquet...")
            ds = HFDataset.from_parquet(str(parquet_path))
            if max_samples and len(ds) > max_samples:
                ds = ds.shuffle(seed=42).select(range(max_samples))
            return ds

        # First load: parse JSON → Arrow → Parquet cache
        print(f"    Parsing llava_instruct_150k.json...")
        try:
            import orjson
            with open(json_path, "rb") as f:
                records = orjson.loads(f.read())
            print(f"    Parsed {len(records):,} records with orjson")
        except ImportError:
            with open(json_path, "r") as f:
                records = json.load(f)
            print(f"    Parsed {len(records):,} records with json")

        for r in records:
            if "conversations" in r and isinstance(r["conversations"], list):
                r["conversations"] = json.dumps(r["conversations"])

        print(f"    Converting to Arrow dataset...")
        df = pd.DataFrame(records)
        del records
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).replace("None", pd.NA).replace("nan", pd.NA)
        ds = HFDataset.from_pandas(df)
        del df

        print(f"    Caching as Parquet at {parquet_path}...")
        ds.to_parquet(str(parquet_path))
        print(f"    Parquet cache saved")

        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds

    def get_language(self, example: dict) -> str:
        return "en"

    def to_conversations(self, example: dict) -> list[dict]:
        return self._parse_conversations(example["conversations"])


# Mapping from ISO 639-3 / Aya full language names to 2-letter codes.
# Aya Dataset uses full names in 'language' and 3-letter in 'language_code'.
_AYA_LANG_MAP = {
    # Full names → 2-letter
    "English": "en", "French": "fr", "Spanish": "es", "Portuguese": "pt",
    "Italian": "it", "German": "de", "Dutch": "nl", "Russian": "ru",
    "Polish": "pl", "Ukrainian": "uk", "Czech": "cs", "Romanian": "ro",
    "Hungarian": "hu", "Finnish": "fi", "Swedish": "sv", "Danish": "da",
    "Norwegian": "no", "Greek": "el", "Turkish": "tr", "Arabic": "ar",
    "Standard Arabic": "ar", "Moroccan Arabic": "ar", "Egyptian Arabic": "ar",
    "Persian": "fa", "Hebrew": "he", "Hindi": "hi", "Bengali": "bn",
    "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa", "Tamil": "ta",
    "Telugu": "te", "Urdu": "ur", "Nepali": "ne",
    "Chinese": "zh", "Simplified Chinese": "zh", "Traditional Chinese": "zh",
    "Japanese": "ja", "Korean": "ko",
    "Vietnamese": "vi", "Thai": "th", "Indonesian": "id", "Malay": "ms",
    "Tagalog": "tl", "Javanese": "jv",
    "Lao": "lo", "Burmese": "my", "Myanmar": "my", "Khmer": "km", "Cambodian": "km",
    "Swahili": "sw", "Yoruba": "yo", "Hausa": "ha", "Igbo": "ig",
    "Amharic": "am", "Shona": "sn", "Zulu": "zu", "Xhosa": "xh",
    "Wolof": "wo", "Malagasy": "mg",
    "Catalan": "ca", "Galician": "gl", "Basque": "eu", "Welsh": "cy",
    "Irish": "ga", "Croatian": "hr", "Serbian": "sr", "Slovak": "sk",
    "Slovenian": "sl", "Bulgarian": "bg", "Latvian": "lv", "Lithuanian": "lt",
    "Estonian": "et", "Maltese": "mt",
    # ISO 639-3 → 2-letter (language_code field)
    "eng": "en", "fra": "fr", "spa": "es", "por": "pt", "ita": "it",
    "deu": "de", "nld": "nl", "rus": "ru", "pol": "pl", "ukr": "uk",
    "ces": "cs", "ron": "ro", "hun": "hu", "fin": "fi", "swe": "sv",
    "dan": "da", "nor": "no", "ell": "el", "tur": "tr", "arb": "ar",
    "ary": "ar", "arz": "ar", "fas": "fa", "heb": "he",
    "hin": "hi", "ben": "bn", "mar": "mr", "guj": "gu", "pan": "pa",
    "tam": "ta", "tel": "te", "urd": "ur", "nep": "ne",
    "zho": "zh", "jpn": "ja", "kor": "ko",
    "vie": "vi", "tha": "th", "ind": "id", "msa": "ms", "tgl": "tl",
    "lao": "lo", "mya": "my", "khm": "km",
    "jav": "jv", "swa": "sw", "yor": "yo", "hau": "ha", "ibo": "ig",
    "amh": "am", "sna": "sn", "zul": "zu", "xho": "xh", "wol": "wo",
    "mlg": "mg", "cat": "ca", "glg": "gl", "eus": "eu", "cym": "cy",
    "gle": "ga", "hrv": "hr", "srp": "sr", "slk": "sk", "slv": "sl",
    "bul": "bg", "lav": "lv", "lit": "lt", "est": "et", "mlt": "mt",
}


class AyaDatasetSource(DatasetSource):
    """CohereForAI/aya_dataset — multilingual text-only instruction data.

    Verified schema: {inputs (str), targets (str), language (full name str),
                      language_code (ISO 639-3 str), annotation_type, user_id}
    Preserves text-only LLM capabilities during multimodal training.
    """
    name = "aya_text"
    hf_dataset_id = "CohereForAI/aya_dataset"

    def get_language(self, example: dict) -> str:
        # Try language_code (ISO 639-3) first, fall back to full name
        code = example.get("language_code", "")
        if code in _AYA_LANG_MAP:
            return _AYA_LANG_MAP[code]
        name = example.get("language", "")
        return _AYA_LANG_MAP.get(name, name.lower()[:2])

    def to_conversations(self, example: dict) -> list[dict]:
        return [
            {"from": "human", "value": example["inputs"]},
            {"from": "gpt", "value": example["targets"]},
        ]

    def get_image(self, example: dict) -> Image.Image | None:
        return None  # Text-only dataset



# ---------------------------------------------------------------------------
# ALM-Bench language name → 2-letter ISO code mapping
# ---------------------------------------------------------------------------

_ALM_LANG_MAP = {
    "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar",
    "Armenian": "hy", "Assamese": "as", "Azerbaijani": "az",
    "Basque": "eu", "Belarusian": "be", "Bengali": "bn", "Bhojpuri": "bh",
    "Bosnian": "bs", "Bulgarian": "bg",
    "Catalan": "ca", "Cebuano": "ceb", "Chinese (Simplified)": "zh",
    "Chinese (Traditional)": "zh", "Croatian": "hr", "Czech": "cs",
    "Danish": "da", "Dutch": "nl",
    "Egyptian Arabic": "ar", "Emirati Arabic": "ar", "English": "en",
    "Estonian": "et",
    "Filipino": "tl", "Finnish": "fi", "French": "fr",
    "Galician": "gl", "Georgian": "ka", "German": "de", "Greek": "el",
    "Gujarati": "gu",
    "Hausa": "ha", "Hawaiian": "haw", "Hebrew": "he", "Hindi": "hi",
    "Hungarian": "hu",
    "Icelandic": "is", "Igbo": "ig", "Indonesian": "id", "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja", "Javanese": "jv",
    "Kannada": "kn", "Kazakh": "kk", "Khmer": "km", "Kinyarwanda": "rw",
    "Korean": "ko", "Kurdish": "ku", "Kyrgyz": "ky",
    "Lao": "lo", "Latin": "la", "Latvian": "lv", "Lithuanian": "lt",
    "Luxembourgish": "lb",
    "Macedonian": "mk", "Malagasy": "mg", "Malay": "ms", "Malayalam": "ml",
    "Maltese": "mt", "Marathi": "mr", "Mongolian": "mn",
    "Myanmar (Burmese)": "my",
    "Nepali": "ne", "Norwegian": "no",
    "Odia (Oriya)": "or",
    "Pashto": "ps", "Persian": "fa", "Polish": "pl", "Portuguese": "pt",
    "Punjabi": "pa",
    "Romanian": "ro", "Russian": "ru",
    "Sanskrit": "sa", "Saudi Arabic": "ar", "Scots Gaelic": "gd",
    "Serbian": "sr", "Shona": "sn", "Sindhi": "sd", "Sinhala": "si",
    "Slovak": "sk", "Slovenian": "sl", "Somali": "so", "Spanish": "es",
    "Sundanese": "su", "Swahili": "sw", "Swedish": "sv",
    "Tajik": "tg", "Tamil": "ta", "Telugu": "te", "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk", "Urdu": "ur", "Uyghur": "ug", "Uzbek": "uz",
    "Vietnamese": "vi",
    "Welsh": "cy",
    "Yiddish": "yi", "Yoruba": "yo",
}


# ---------------------------------------------------------------------------
# MVL-SIB language code → 2-letter ISO code mapping
# Uses SIB-200 codes like "amh_Ethi" → "am"
# ---------------------------------------------------------------------------

_MVL_SIB_LANG_MAP = {
    "amh_Ethi": "am", "bul_Cyrl": "bg", "cat_Latn": "ca", "cym_Latn": "cy",
    "dan_Latn": "da", "est_Latn": "et", "eus_Latn": "eu", "fin_Latn": "fi",
    "gle_Latn": "ga", "glg_Latn": "gl", "hrv_Latn": "hr", "hun_Latn": "hu",
    "khm_Khmr": "km", "lao_Laoo": "lo", "lit_Latn": "lt", "lvs_Latn": "lv",
    "plt_Latn": "mg", "mlt_Latn": "mt", "mya_Mymr": "my", "npi_Deva": "ne",
    "pan_Guru": "pa", "slk_Latn": "sk", "slv_Latn": "sl", "sna_Latn": "sn",
    "srp_Cyrl": "sr", "swe_Latn": "sv", "tgl_Latn": "tl", "wol_Latn": "wo",
    # Also map non-tier2 languages that Tiny Aya supports
    "arb_Arab": "ar", "ben_Beng": "bn", "ces_Latn": "cs", "deu_Latn": "de",
    "ell_Grek": "el", "eng_Latn": "en", "fra_Latn": "fr", "guj_Gujr": "gu",
    "hau_Latn": "ha", "heb_Hebr": "he", "hin_Deva": "hi", "ind_Latn": "id",
    "ibo_Latn": "ig", "ita_Latn": "it", "jpn_Jpan": "ja", "jav_Latn": "jv",
    "kor_Hang": "ko", "zsm_Latn": "ms", "mar_Deva": "mr", "nld_Latn": "nl",
    "nob_Latn": "no", "pes_Arab": "fa", "pol_Latn": "pl", "por_Latn": "pt",
    "ron_Latn": "ro", "rus_Cyrl": "ru", "spa_Latn": "es", "swh_Latn": "sw",
    "tam_Taml": "ta", "tel_Telu": "te", "tha_Thai": "th", "tur_Latn": "tr",
    "ukr_Cyrl": "uk", "urd_Arab": "ur", "vie_Latn": "vi", "xho_Latn": "xh",
    "yor_Latn": "yo", "zho_Hans": "zh", "zul_Latn": "zu",
}


class ALMBenchSource(DatasetSource):
    """MBZUAI/ALM-Bench — 100-language multimodal VQA benchmark (22K+ samples).

    Schema: {file_name (PIL Image), ID, Language (full name str), Category,
             Question_Type, English_Question, English_Answer,
             Translated_Question, Translated_Answer, Image_Url}

    Images are embedded as PIL Image objects in the "file_name" column.
    Each sample is a visual QA pair in one of 100 languages.
    """
    name = "alm_bench"
    hf_dataset_id = "MBZUAI/ALM-Bench"
    split = "test"

    def get_language(self, example: dict) -> str:
        lang_name = example.get("Language", "English")
        return _ALM_LANG_MAP.get(lang_name, "en")

    def to_conversations(self, example: dict) -> list[dict]:
        question = example.get("Translated_Question") or example.get("English_Question", "")
        answer = example.get("Translated_Answer") or example.get("English_Answer", "")
        return [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ]

    def get_image(self, example: dict) -> Image.Image | None:
        img = example.get("file_name")
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        return None


class MVLSIBSource(DatasetSource):
    """WueNLP/mvl-sib — Multilingual Visual Language SIB-200 (197 languages).

    The dataset stores topic-classification texts per language in TSV files,
    paired with shared topic images. Each language has ~701 train samples
    across 7 topic categories, each mapped to a representative image.

    Schema (TSV): index_id, category, text
    Images: data/images/sib200/{category}_{idx}.jpg (shared across languages)

    We convert each sample into visual QA: image + "What topic does this text
    describe?" / text, creating multimodal instruction pairs.
    """
    name = "mvl_sib"
    hf_dataset_id = "WueNLP/mvl-sib"

    # Map SIB-200 categories to image filenames (10 images per category)
    _CATEGORIES = [
        "entertainment", "geography", "health", "politics",
        "science/technology", "sports", "travel",
    ]

    def __init__(self, data_dir: str | None = None):
        super().__init__(data_dir)
        self._images_cache: dict[str, Image.Image] = {}

    def _prefetch_images(self, cache_dir: str | None = None) -> None:
        """Pre-download all 70 topic images (7 categories x 10 indices) at init time.

        Images are small and finite — downloading them all upfront avoids
        per-sample HTTP requests during training.
        """
        from huggingface_hub import hf_hub_download

        print("    Pre-fetching MVL-SIB topic images...")
        fetched, failed = 0, 0
        for cat in self._CATEGORIES:
            cat_clean = cat.replace("/", "_").replace(" ", "_").lower()
            for idx in range(10):
                img_key = f"{cat_clean}_{idx}.jpg"
                if img_key in self._images_cache:
                    continue
                try:
                    img_path = hf_hub_download(
                        self.hf_dataset_id,
                        f"data/images/sib200/{img_key}",
                        repo_type="dataset",
                        cache_dir=cache_dir,
                    )
                    self._images_cache[img_key] = Image.open(img_path).convert("RGB")
                    fetched += 1
                except Exception:
                    failed += 1
        print(f"    Pre-fetched {fetched} images, {failed} missing")

    def load(self, cache_dir: str | None = None, max_samples: int | None = None) -> HFDataset:
        """Download and parse MVL-SIB TSV files for each requested language.

        Returns a HuggingFace Dataset with columns:
        {text, category, lang_code, sib_lang_dir}
        """
        from huggingface_hub import HfApi, hf_hub_download
        import csv

        # Pre-download all topic images so get_image() is purely local
        self._prefetch_images(cache_dir=cache_dir)

        api = HfApi()
        all_dirs = [f.path.split("/")[-1] for f in
                    api.list_repo_tree(self.hf_dataset_id, repo_type="dataset",
                                       path_in_repo="data/sib200")]

        records = []
        for lang_dir in all_dirs:
            # Map to 2-letter code; skip if not a Tiny Aya language
            iso2 = _MVL_SIB_LANG_MAP.get(lang_dir)
            if iso2 is None:
                continue

            try:
                tsv_path = hf_hub_download(
                    self.hf_dataset_id,
                    f"data/sib200/{lang_dir}/train.tsv",
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
                with open(tsv_path) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        records.append({
                            "text": row["text"],
                            "category": row["category"],
                            "lang_code": iso2,
                            "sib_lang_dir": lang_dir,
                        })
            except Exception:
                continue

        ds = HFDataset.from_list(records)
        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds

    def get_language(self, example: dict) -> str:
        return example.get("lang_code", "en")

    def to_conversations(self, example: dict) -> list[dict]:
        text = example.get("text", "")
        category = example.get("category", "")
        return [
            {"from": "human", "value": f"<image>\nWhat is described in this text? Classify the topic.\n\n{text}"},
            {"from": "gpt", "value": category},
        ]

    def get_image(self, example: dict) -> Image.Image | None:
        """Return a representative topic image from the pre-fetched cache."""
        category = example.get("category", "")
        cat_clean = category.replace("/", "_").replace(" ", "_").lower()
        text_hash = hash(example.get("text", "")) % 10
        img_key = f"{cat_clean}_{text_hash}.jpg"
        return self._images_cache.get(img_key)

class BloomLMSource(DatasetSource):
    """sil-ai/bloom-lm — Multilingual text data across 364 languages.

    Schema: JSON dict of {lang_code: [{text: str}, ...]}
    Text-only dataset with broad low-resource language coverage.
    Supplements gaps in multimodal data with text-only instruction data.
    """
    name = "bloom_lm"
    hf_dataset_id = "sil-ai/bloom-lm"

    def load(self, cache_dir: str | None = None, max_samples: int | None = None) -> HFDataset:
        """Download and parse bloom-lm JSON into a flat HF Dataset."""
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            self.hf_dataset_id,
            "bloom_lm_train.json",
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        with open(path) as f:
            raw = json.load(f)

        records = []
        for lang_code, texts in raw.items():
            # Map to 2-letter codes
            iso2 = _AYA_LANG_MAP.get(lang_code, lang_code)
            if len(iso2) > 3:
                iso2 = lang_code  # keep original if no mapping
            for item in texts:
                records.append({
                    "text": item.get("text", ""),
                    "lang_code": iso2,
                })

        ds = HFDataset.from_list(records)
        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds

    def get_language(self, example: dict) -> str:
        return example.get("lang_code", "en")

    def to_conversations(self, example: dict) -> list[dict]:
        text = example.get("text", "")
        # Create a simple instruction from raw text
        if len(text) > 200:
            return [
                {"from": "human", "value": "Summarize or continue the following text:"},
                {"from": "gpt", "value": text},
            ]
        return [
            {"from": "human", "value": "Complete or respond to the following:"},
            {"from": "gpt", "value": text},
        ]

    def get_image(self, example: dict) -> Image.Image | None:
        return None  # Text-only dataset


# ---------------------------------------------------------------------------
# WIT language code mapping (WIT uses BCP-47 / wiki codes)
# ---------------------------------------------------------------------------

_WIT_LANG_MAP = {
    "en": "en", "de": "de", "fr": "fr", "es": "es", "ru": "ru",
    "it": "it", "pl": "pl", "ja": "ja", "nl": "nl", "sv": "sv",
    "uk": "uk", "pt": "pt", "zh": "zh", "zh-TW": "zh",
    "ca": "ca", "cs": "cs", "hu": "hu", "ar": "ar", "vi": "vi",
    "no": "no", "fa": "fa", "iw": "he", "he": "he",  # WIT uses "iw" for Hebrew
    "sr": "sr", "ro": "ro", "ko": "ko", "eu": "eu", "bg": "bg",
    "tr": "tr", "fi": "fi", "da": "da", "el": "el", "sk": "sk",
    "hr": "hr", "lt": "lt", "et": "et", "gl": "gl", "sl": "sl",
    "lv": "lv", "cy": "cy", "ga": "ga", "id": "id", "th": "th",
    "hi": "hi", "bn": "bn", "ta": "ta", "te": "te", "ur": "ur",
    "mr": "mr", "gu": "gu", "pa": "pa", "ne": "ne", "ms": "ms",
    "my": "my", "sw": "sw", "mg": "mg", "jv": "jv",
    "mt": "mt", "af": "af",
}


class WITSource(DatasetSource):
    """wikimedia/wit_base — Wikipedia Image Text (~6.5M image-text pairs, 105 languages).

    Each row contains an image (embedded) and multilingual captions/descriptions
    from Wikipedia. The `wit_features` column contains per-language metadata
    including page descriptions, section descriptions, and page titles.

    Schema: {image (PIL Image via dict), image_url, wit_features: {language[],
             context_page_description[], page_title[], ...}}

    We explode each row into one sample per language, using the
    context_page_description or page_title as the caption text.
    """
    name = "wit"
    hf_dataset_id = "wikimedia/wit_base"
    split = "train"

    def __init__(self, data_dir: str | None = None):
        super().__init__(data_dir)
        self._shard_count = 330  # 330 parquet shards

    def load(self, cache_dir: str | None = None, max_samples: int | None = None) -> HFDataset:
        """Load WIT and explode multilingual features into flat rows.

        WIT is ~300GB across 330 shards. We stream a subset of shards
        and flatten the nested wit_features into one row per (image, language).
        """
        from huggingface_hub import hf_hub_download

        # Determine how many shards to load based on max_samples
        # Each shard has ~20K rows, each row has ~10 languages = ~200K (lang, image) pairs
        if max_samples:
            n_shards = max(1, min(self._shard_count, max_samples // 200_000 + 1))
        else:
            n_shards = 10  # default: 10 shards ≈ 2M samples

        records = []
        for shard_idx in range(n_shards):
            shard_file = f"data/train-{shard_idx:05d}-of-00330.parquet"
            try:
                path = hf_hub_download(
                    self.hf_dataset_id, shard_file,
                    repo_type="dataset", cache_dir=cache_dir,
                )
            except Exception:
                continue

            df = pd.read_parquet(path)
            print(f"    WIT shard {shard_idx}: {len(df)} rows")

            for _, row in df.iterrows():
                wf = row.get("wit_features")
                if wf is None:
                    continue
                languages = wf.get("language", [])
                descriptions = wf.get("context_page_description", [])
                titles = wf.get("page_title", [])
                image_bytes = row.get("image")

                for j, lang_code in enumerate(languages):
                    iso2 = _WIT_LANG_MAP.get(lang_code)
                    if iso2 is None:
                        continue
                    # Pick the best available text: description > title
                    text = None
                    if j < len(descriptions) and descriptions[j]:
                        text = descriptions[j]
                    elif j < len(titles) and titles[j]:
                        text = titles[j]
                    if not text:
                        continue

                    records.append({
                        "text": text[:1024],  # cap long Wikipedia paragraphs
                        "lang_code": iso2,
                        "image_bytes": image_bytes,
                    })

            del df
            if max_samples and len(records) >= max_samples:
                break

        print(f"    WIT total: {len(records):,} (lang, image) pairs from {n_shards} shards")
        ds = HFDataset.from_list(records[:max_samples] if max_samples else records)
        return ds

    def get_language(self, example: dict) -> str:
        return example.get("lang_code", "en")

    def to_conversations(self, example: dict) -> list[dict]:
        text = example.get("text", "")
        return [
            {"from": "human", "value": "<image>\nDescribe this image."},
            {"from": "gpt", "value": text},
        ]

    def get_image(self, example: dict) -> Image.Image | None:
        img_data = example.get("image_bytes")
        if img_data is None:
            return None
        if isinstance(img_data, dict) and "bytes" in img_data:
            import io
            try:
                return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            except Exception:
                return None
        if isinstance(img_data, Image.Image):
            return img_data.convert("RGB")
        return None


class BloomCaptioningSource(DatasetSource):
    """sil-ai/bloom-captioning — Image captioning in 400+ languages (10-100K samples).

    Expert-generated image captions across extremely low-resource languages.
    Covers am, km, lo, my, ne, pa, tl, ha, xh, yo, zu + many more.
    NOTE: This is a gated dataset — requires HF token with access.

    Schema: {image_url (str), caption (str), language (ISO 639-3 code)}
    Images are downloaded from URLs and cached to disk on first access.
    """
    name = "bloom_captioning"
    hf_dataset_id = "sil-ai/bloom-captioning"
    split = "test"  # use test split (default available)

    def __init__(self, data_dir: str | None = None):
        super().__init__(data_dir)
        self._image_cache_dir: Path | None = None
        self._failed_urls: set[str] = set()

    def load(self, cache_dir: str | None = None, max_samples: int | None = None) -> HFDataset:
        """Load bloom-captioning dataset."""
        # Set up a disk cache directory for downloaded images
        base = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        self._image_cache_dir = base / "bloom_captioning_images"
        self._image_cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            ds = load_dataset(self.hf_dataset_id, split=self.split,
                              cache_dir=cache_dir, trust_remote_code=True)
        except Exception as e:
            print(f"    bloom-captioning: could not load (gated?): {e}")
            print(f"    Falling back to empty dataset")
            return HFDataset.from_list([])

        if max_samples and len(ds) > max_samples:
            ds = ds.shuffle(seed=42).select(range(max_samples))
        return ds

    def get_language(self, example: dict) -> str:
        code = example.get("language", "")
        # bloom-captioning uses ISO 639-3 codes
        return _AYA_LANG_MAP.get(code, code[:2] if len(code) >= 2 else "en")

    def to_conversations(self, example: dict) -> list[dict]:
        caption = example.get("caption", "")
        return [
            {"from": "human", "value": "<image>\nDescribe this image."},
            {"from": "gpt", "value": caption},
        ]

    def get_image(self, example: dict) -> Image.Image | None:
        """Load image from disk cache, downloading from URL only on first access."""
        import hashlib
        url = example.get("image_url", "")
        if not url or url in self._failed_urls:
            return None

        # Use URL hash as the cache filename
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        cache_path = self._image_cache_dir / f"{url_hash}.jpg" if self._image_cache_dir else None

        # Try loading from disk cache first
        if cache_path and cache_path.exists():
            try:
                return Image.open(cache_path).convert("RGB")
            except Exception:
                cache_path.unlink(missing_ok=True)

        # Download and cache to disk
        try:
            import requests
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                import io
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                if cache_path:
                    img.save(cache_path, "JPEG")
                return img
            else:
                self._failed_urls.add(url)
        except Exception:
            self._failed_urls.add(url)
        return None


# `Regis`try of available sources
DATASET_SOURCES = {
    "pangea_ins": PangeaInsSource,
    "palo": PaloSource,
    "llava_instruct": LLaVAInstructSource,
    "aya_text": AyaDatasetSource,
    "alm_bench": ALMBenchSource,
    "mvl_sib": MVLSIBSource,
    "bloom_lm": BloomLMSource,
    "wit": WITSource,
    "bloom_captioning": BloomCaptioningSource,
}


# ---------------------------------------------------------------------------
# Unified multilingual dataset
# ---------------------------------------------------------------------------

class MultilingualInstructDataset(torch.utils.data.Dataset):
    """Balanced multilingual instruction dataset with temperature-based sampling.

    Loads data from multiple sources, tags each example with its language,
    then builds a balanced index using temperature sampling that respects
    the target English/multilingual ratio.

    Compatible with the existing collate_fn from pipeline.data.
    """

    ROLE_MAP = {"human": "user", "gpt": "assistant"}

    def __init__(
        self,
        config: TinyAyaVisionConfig,
        sources: list[dict],
        total_samples: int = 500_000,
        english_ratio: float = 0.4,
        temperature: float = 5.0,
        max_seq_len: int = 2048,
        cache_dir: str | None = None,
        seed: int = 42,
        allow_upsampling: bool = True,
        max_upsample_factor: float = 5.0,
    ):
        """
        Args:
            config: Model configuration for processor initialization.
            sources: List of source configs, each a dict with:
                - name: str (key in DATASET_SOURCES)
                - max_samples: int | None (cap per source)
                - languages: list[str] | None (filter to specific languages)
            total_samples: Total dataset size after mixing.
            english_ratio: Fraction of total allocated to English (default 0.4).
            temperature: Temperature for non-English language sampling.
            max_seq_len: Maximum sequence length for tokenization.
            cache_dir: HuggingFace cache directory.
            seed: Random seed for reproducible sampling.
            allow_upsampling: Allow low-resource languages to be sampled with
                replacement beyond their available count.
            max_upsample_factor: Maximum factor by which a language can be
                upsampled (e.g. 5.0 = up to 5× its available data).
        """
        self.processor = TinyAyaVisionProcessor(config=config)
        self.max_seq_len = max_seq_len
        self.seed = seed

        # Cache special token IDs for label masking
        self._chatbot_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|CHATBOT_TOKEN|>"
        )
        self._end_turn_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|END_OF_TURN_TOKEN|>"
        )

        # Load all sources into a unified list of (source, raw_example) tuples
        # grouped by language
        print("Loading multilingual dataset sources...")
        self._lang_examples: dict[str, list[tuple[DatasetSource, dict]]] = defaultdict(list)
        self._sources_cache: dict[str, DatasetSource] = {}

        for src_cfg in sources:
            src_name = src_cfg["name"]
            src_cls = DATASET_SOURCES[src_name]
            src_data_dir = src_cfg.get("data_dir")
            src = src_cls(data_dir=src_data_dir)
            self._sources_cache[src_name] = src

            max_per_source = src_cfg.get("max_samples")
            filter_langs = set(src_cfg.get("languages", []))

            print(f"  Loading {src_name} ({src.hf_dataset_id})...")
            raw_ds = src.load(cache_dir=cache_dir, max_samples=max_per_source)
            print(f"    Loaded {len(raw_ds)} examples")

            for i in range(len(raw_ds)):
                example = raw_ds[i]
                lang = src.get_language(example)
                if filter_langs and lang not in filter_langs:
                    continue
                self._lang_examples[lang].append((src, example))

        # Report language distribution before mixing
        lang_counts = {l: len(exs) for l, exs in self._lang_examples.items()}
        total_available = sum(lang_counts.values())
        print(f"\nAvailable data: {total_available} examples across {len(lang_counts)} languages")
        for lang in sorted(lang_counts, key=lang_counts.get, reverse=True)[:20]:
            family = LANG_TO_FAMILY.get(lang, "unknown")
            print(f"  {lang} ({family}): {lang_counts[lang]:,}")
        if len(lang_counts) > 20:
            print(f"  ... and {len(lang_counts) - 20} more languages")

        # Compute balanced sampling counts
        target_per_lang = compute_sampling_indices(
            lang_counts=lang_counts,
            total_samples=total_samples,
            temperature=temperature,
            english_ratio=english_ratio,
            seed=seed,
            allow_upsampling=allow_upsampling,
            max_upsample_factor=max_upsample_factor,
        )

        # Build the final index: list of (source, example) tuples
        rng = np.random.RandomState(seed)
        self._examples: list[tuple[DatasetSource, dict]] = []
        actual_counts: dict[str, int] = {}

        for lang, n_target in target_per_lang.items():
            pool = self._lang_examples[lang]
            if not pool or n_target == 0:
                continue
            indices = rng.choice(len(pool), size=n_target, replace=(n_target > len(pool)))
            selected = [pool[i] for i in indices]
            self._examples.extend(selected)
            actual_counts[lang] = n_target

        # Shuffle the combined dataset
        rng.shuffle(self._examples)

        # Report final mix
        total_mixed = len(self._examples)
        en_count = actual_counts.get("en", 0)
        non_en_count = total_mixed - en_count
        print(f"\nFinal mixed dataset: {total_mixed} examples")
        print(f"  English: {en_count} ({100*en_count/max(total_mixed,1):.1f}%)")
        print(f"  Non-English: {non_en_count} ({100*non_en_count/max(total_mixed,1):.1f}%)")
        print(f"  Languages represented: {len(actual_counts)}")
        for lang in sorted(actual_counts, key=actual_counts.get, reverse=True)[:15]:
            pct = 100 * actual_counts[lang] / total_mixed
            print(f"    {lang}: {actual_counts[lang]:,} ({pct:.1f}%)")

        # Free raw grouped data to save memory
        del self._lang_examples

    def _to_chat_messages(self, conversations: list[dict]) -> list[dict]:
        """Convert LLaVA-style conversations to chat template messages."""
        messages = []
        for turn in conversations:
            role = self.ROLE_MAP.get(turn["from"], turn["from"])
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

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mask labels so loss is only on assistant responses."""
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
        return len(self._examples)

    def __getitem__(self, idx):
        try:
            return self._get_example(idx)
        except Exception:
            # On any failure, return a minimal text-only dummy so the
            # DataLoader worker doesn't hang and desync NCCL ranks.
            return self._fallback_example()

    def _get_example(self, idx):
        src, example = self._examples[idx]

        conversations = src.to_conversations(example)
        image = src.get_image(example)
        has_image = image is not None

        # If image failed to load but text has <image> tags, strip them to
        # prevent image placeholder tokens from appearing in input_ids
        # without corresponding pixel_values.
        if not has_image:
            for turn in conversations:
                turn["value"] = (
                    turn["value"]
                    .replace("<image>\n", "")
                    .replace("\n<image>", "")
                    .replace("<image>", "")
                )

        messages = self._to_chat_messages(conversations)

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        processed = self.processor(
            text=full_text,
            images=image if has_image else None,
            truncation=True,
            max_length=self.max_seq_len,
        )

        input_ids = processed["input_ids"].squeeze(0)
        attention_mask = processed["attention_mask"].squeeze(0)
        labels = self._build_labels(input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": processed["pixel_values"].squeeze(0) if "pixel_values" in processed else None,
            "labels": labels,
        }
        return result

    def _fallback_example(self):
        """Return a minimal text-only example so training doesn't crash."""
        dummy_text = self.processor.apply_chat_template(
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            tokenize=False, add_generation_prompt=False,
        )
        processed = self.processor(text=dummy_text, truncation=True, max_length=self.max_seq_len)
        input_ids = processed["input_ids"].squeeze(0)
        attention_mask = processed["attention_mask"].squeeze(0)
        labels = self._build_labels(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": None,
            "labels": labels,
        }
