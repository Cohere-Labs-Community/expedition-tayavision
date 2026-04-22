"""Tests for the evaluation harness setup."""

import yaml
import pytest
from pathlib import Path

from evaluation.m_arena_hard import LANGUAGES, load_arena_hard

TASKS_DIR = Path(__file__).parent.parent / "evaluation" / "tasks"


class TestMArenaHard:
    """Tests for the m-ArenaHard data loading scaffold."""

    def test_languages_list(self):
        """All 23 languages should be listed."""
        assert len(LANGUAGES) == 23
        assert "en" in LANGUAGES
        assert "ar" in LANGUAGES
        assert "zh" in LANGUAGES

    def test_load_english(self):
        """English subset loads correctly."""
        ds = load_arena_hard("en")
        assert len(ds) > 0
        sample = ds[0]
        assert "question_id" in sample
        assert "prompt" in sample

    def test_load_arabic(self):
        """Arabic subset loads correctly."""
        ds = load_arena_hard("ar")
        assert len(ds) > 0

    def test_invalid_language_raises(self):
        """Invalid language code raises ValueError."""
        with pytest.raises(ValueError, match="not in m-ArenaHard"):
            load_arena_hard("xx")


class TestCVQATask:
    """Tests for the CVQA lm-eval task configuration."""

    def test_utils_import(self):
        """CVQA utils module imports without errors."""
        from evaluation.tasks.cvqa.utils import (
            cvqa_doc_to_image,
            cvqa_doc_to_target,
            cvqa_doc_to_text,
            cvqa_process_results,
        )

        assert callable(cvqa_doc_to_image)
        assert callable(cvqa_doc_to_text)
        assert callable(cvqa_doc_to_target)
        assert callable(cvqa_process_results)

    def test_doc_to_text_format(self):
        """CVQA prompt formatting produces expected format."""
        from evaluation.tasks.cvqa.utils import cvqa_doc_to_text

        doc = {
            "Question": "What color is the sky?",
            "Options": ["Red", "Blue", "Green", "Yellow"],
            "Label": 1,
        }
        text = cvqa_doc_to_text(doc)
        assert "<image>" in text
        assert "A." in text
        assert "B." in text
        assert "Blue" in text

    def test_process_results_correct(self):
        """Correct answer scores 1.0."""
        from evaluation.tasks.cvqa.utils import cvqa_process_results

        doc = {"Label": 1}  # B
        results = ["B"]
        score = cvqa_process_results(doc, results)
        assert score["exact_match"] == 1.0

    def test_process_results_incorrect(self):
        """Incorrect answer scores 0.0."""
        from evaluation.tasks.cvqa.utils import cvqa_process_results

        doc = {"Label": 1}  # B
        results = ["A"]
        score = cvqa_process_results(doc, results)
        assert score["exact_match"] == 0.0

    def test_load_cvqa_uses_metadata_chunk_bounds(self, monkeypatch):
        """CVQA custom dataset loader slices using lm-eval metadata."""
        from datasets import Dataset
        from evaluation.tasks.cvqa import utils

        requested_splits = []

        def fake_load_dataset(path, split):
            requested_splits.append((path, split))
            return Dataset.from_list([])

        monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

        ds = utils.load_cvqa(cvqa_chunk_start=10, cvqa_chunk_end=20)

        assert "test" in ds
        assert requested_splits == [("afaji/cvqa", "test[10:20]")]


class TestXMMMUTask:
    """Tests for the xMMMU lm-eval task configuration."""

    def test_utils_import(self):
        """xMMMU utils module imports without errors."""
        from evaluation.tasks.xmmmu.utils import (
            xmmmu_doc_to_image,
            xmmmu_doc_to_text,
            xmmmu_process_results,
        )

        assert callable(xmmmu_doc_to_image)
        assert callable(xmmmu_doc_to_text)
        assert callable(xmmmu_process_results)

    def test_doc_to_image_extracts_non_none(self):
        """Only non-None images are extracted."""
        from evaluation.tasks.xmmmu.utils import xmmmu_doc_to_image

        doc = {
            "image_1": "img1",
            "image_2": None,
            "image_3": "img3",
            **{f"image_{i}": None for i in range(4, 8)},
        }
        images = xmmmu_doc_to_image(doc)
        assert len(images) == 2
        assert images[0] == "img1"
        assert images[1] == "img3"

    def test_doc_to_text_replaces_image_markers(self):
        """<image N> markers are replaced with <image>."""
        from evaluation.tasks.xmmmu.utils import xmmmu_doc_to_text

        doc = {
            "question": "<image 1> What is shown?",
            "options": "['A thing', 'B thing', 'C thing', 'D thing']",
            "question_type": "multiple-choice",
        }
        text = xmmmu_doc_to_text(doc)
        assert "<image 1>" not in text
        assert "<image>" in text
        assert "A." in text

    def test_doc_to_text_open_ended(self):
        """Open-ended questions use a different prompt."""
        from evaluation.tasks.xmmmu.utils import xmmmu_doc_to_text

        doc = {
            "question": "<image 1> Compute the value.",
            "options": "[]",
            "question_type": "open",
        }
        text = xmmmu_doc_to_text(doc)
        assert "single word or phrase" in text
        assert "A." not in text

    def test_process_results_mcq_correct(self):
        """Correct MCQ answer scores 1.0."""
        from evaluation.tasks.xmmmu.utils import xmmmu_process_results

        doc = {
            "answer": "B",
            "question_type": "multiple-choice",
            "options": "['opt1', 'opt2', 'opt3', 'opt4']",
        }
        results = ["B"]
        score = xmmmu_process_results(doc, results)
        assert score["exact_match"] == 1.0

    def test_process_results_open_numeric(self):
        """Open-ended numeric answer scores correctly."""
        from evaluation.tasks.xmmmu.utils import xmmmu_process_results

        doc = {"answer": "1.06", "question_type": "open", "options": "[]"}
        results = ["1.06"]
        assert xmmmu_process_results(doc, results)["exact_match"] == 1.0

        results_wrong = ["2.5"]
        assert xmmmu_process_results(doc, results_wrong)["exact_match"] == 0.0

    def test_process_results_open_string(self):
        """Open-ended string answer with substring match."""
        from evaluation.tasks.xmmmu.utils import xmmmu_process_results

        doc = {"answer": "embryonic", "question_type": "open", "options": "[]"}
        results = ["The answer is embryonic"]
        assert xmmmu_process_results(doc, results)["exact_match"] == 1.0


class TestXMMMUBlindTask:
    """Tests for the xMMMU blind baseline task."""

    def test_utils_import(self):
        """xMMMU blind utils import without errors."""
        from evaluation.tasks.xmmmu.utils import xmmmu_blind_doc_to_text

        assert callable(xmmmu_blind_doc_to_text)

    def test_doc_to_text_strips_image_markers_mcq(self):
        """Image markers are removed for MCQ, options formatted."""
        from evaluation.tasks.xmmmu.utils import xmmmu_blind_doc_to_text

        doc = {
            "question": "<image 1> What is shown in <image 2>?",
            "options": "['Red', 'Blue', 'Green', 'Yellow']",
            "question_type": "multiple-choice",
        }
        text = xmmmu_blind_doc_to_text(doc)
        assert "<image" not in text
        assert "What is shown in" in text
        assert "A." in text
        assert "Blue" in text

    def test_doc_to_text_open_ended(self):
        """Open-ended samples get the correct prompt."""
        from evaluation.tasks.xmmmu.utils import xmmmu_blind_doc_to_text

        doc = {
            "question": "What is 2+2?",
            "options": "[]",
            "question_type": "open",
        }
        text = xmmmu_blind_doc_to_text(doc)
        assert "What is 2+2?" in text
        assert "single word or phrase" in text

MGSM_LANGUAGES = [
    "am", "ar", "bn", "ca", "cs", "cy", "de", "el", "en", "es",
    "eu", "fr", "gl", "gu", "ha", "hu", "ja", "km", "kn", "ko",
    "ky", "lg", "my", "ne", "ru", "si", "sn", "sr", "st", "sw",
    "ta", "te", "th", "ur", "uz", "vi", "wo", "xh", "yo", "zh", "zu",
]


class TestGlobalMGSMTask:
    """Tests for the GlobalMGSM group task configuration."""

    @pytest.fixture
    def config(self):
        yaml_path = TASKS_DIR / "global_mgsm" / "global_mgsm.yaml"
        return yaml.safe_load(yaml_path.read_text())

    def test_yaml_loads(self, config):
        """GlobalMGSM group task YAML loads without errors."""
        assert config is not None

    def test_group_name(self, config):
        """Group is named global_mgsm."""
        assert config["group"] == "global_mgsm"

    def test_all_41_languages_included(self, config):
        """All 41 languages are present as subtasks."""
        task_names = [t["task"] for t in config["task"]]
        for lang in MGSM_LANGUAGES:
            assert f"global_mgsm_direct_{lang}" in task_names

    def test_language_count(self, config):
        """Exactly 41 language subtasks are defined."""
        assert len(config["task"]) == len(MGSM_LANGUAGES)

    def test_aggregate_metric(self, config):
        """exact_match is the aggregate metric."""
        metrics = [m["metric"] for m in config["aggregate_metric_list"]]
        assert "exact_match" in metrics

    def test_per_language_yaml_exists(self):
        """Each language has its own task YAML file."""
        for lang in MGSM_LANGUAGES:
            yaml_path = TASKS_DIR / "global_mgsm" / f"global_mgsm_direct_{lang}.yaml"
            assert yaml_path.exists(), f"Missing YAML for {lang}"
            cfg = yaml.safe_load(yaml_path.read_text())
            assert cfg["task"] == f"global_mgsm_direct_{lang}"
            assert cfg["dataset_name"] == lang

    def test_default_yaml_exists(self):
        """Base _default_yaml template exists and has correct dataset_path."""
        default_path = TASKS_DIR / "global_mgsm" / "_default_yaml"
        assert default_path.exists()
        cfg = yaml.safe_load(default_path.read_text())
        assert cfg["dataset_path"] == "CohereLabs/global-mgsm"
