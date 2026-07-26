"""Tests for the gs:// checkpoint mirror in pipeline.utils.

The bug these guard against: `SAVE_CKPT_DIR` used to reach only the shell
scripts, so `save_checkpoint` wrote to node-local disk while deploy warnings,
queued-resource metadata, and the `qr_watch.sh` durability gate all reported
that checkpoints were durable. A preemption then discarded the run.

Everything here mocks `_gcloud_storage`, so no bucket, credentials, or network
are involved.
"""

from pathlib import Path
from unittest import mock

import pytest

from pipeline import utils


@pytest.fixture
def ckpt_dir(tmp_path):
    """A `/models/<run_id>` lookalike -- the parent name IS the run id."""
    d = tmp_path / "1f7d76da-cb77-43cf-922a-32a9a2891c86"
    d.mkdir()
    return d


def _write(path: Path) -> Path:
    path.write_bytes(b"not a real checkpoint")
    return path


class TestCheckpointRoot:
    def test_unset_is_none(self, monkeypatch):
        monkeypatch.delenv("SAVE_CKPT_DIR", raising=False)
        assert utils.gcs_checkpoint_root() is None

    def test_empty_is_none(self, monkeypatch):
        monkeypatch.setenv("SAVE_CKPT_DIR", "")
        assert utils.gcs_checkpoint_root() is None

    def test_local_path_is_ignored(self, monkeypatch):
        """A node-local path must NOT be treated as durable."""
        monkeypatch.setenv("SAVE_CKPT_DIR", "/models/whatever")
        assert utils.gcs_checkpoint_root() is None

    def test_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts/")
        assert utils.gcs_checkpoint_root() == "gs://bucket/ckpts"


class TestCheckpointStep:
    @pytest.mark.parametrize("name,expected", [
        ("checkpoint_40.pt", 40),
        ("checkpoint_2180.pt", 2180),
        ("gs://b/run/checkpoint_500.pt", 500),
        ("config.json", -1),
    ])
    def test_parsing(self, name, expected):
        assert utils._checkpoint_step(name) == expected

    def test_latest_is_numeric_not_lexicographic(self, ckpt_dir):
        """checkpoint_9 must not beat checkpoint_10."""
        for step in (9, 10, 500):
            _write(ckpt_dir / f"checkpoint_{step}.pt")
        assert utils.find_latest_checkpoint(ckpt_dir).name == "checkpoint_500.pt"


class TestMirror:
    def test_noop_without_env(self, ckpt_dir, monkeypatch):
        monkeypatch.delenv("SAVE_CKPT_DIR", raising=False)
        with mock.patch.object(utils, "_gcloud_storage") as g:
            assert utils.mirror_checkpoint_to_gcs(ckpt_dir / "checkpoint_1.pt") is False
        g.assert_not_called()

    def test_destination_includes_run_id(self, ckpt_dir, monkeypatch):
        """<root>/<run_id>/<file> -- the layout resume reads back from."""
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://tayavision-eu/checkpoints/align")
        src = _write(ckpt_dir / "checkpoint_500.pt")
        with mock.patch.object(utils, "_gcloud_storage",
                               return_value=(True, "", "")) as g:
            assert utils.mirror_checkpoint_to_gcs(src) is True
        g.assert_called_once_with(
            "cp", str(src),
            f"gs://tayavision-eu/checkpoints/align/{ckpt_dir.name}/checkpoint_500.pt",
        )

    def test_failure_warns_but_does_not_raise(self, ckpt_dir, monkeypatch, capsys):
        """A transient bucket error must not kill a multi-hour run -- but it
        must be loud, since the run only *looks* durable afterwards."""
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts")
        src = _write(ckpt_dir / "checkpoint_1.pt")
        with mock.patch.object(utils, "_gcloud_storage",
                               return_value=(False, "", "503 backend error")):
            assert utils.mirror_checkpoint_to_gcs(src) is False
        out = capsys.readouterr().out
        assert "WARNING" in out and "503 backend error" in out


class TestFetch:
    def test_noop_without_env(self, ckpt_dir, monkeypatch):
        monkeypatch.delenv("SAVE_CKPT_DIR", raising=False)
        with mock.patch.object(utils, "_gcloud_storage") as g:
            assert utils.fetch_checkpoint_from_gcs(ckpt_dir) is None
        g.assert_not_called()

    def test_local_checkpoint_wins(self, ckpt_dir, monkeypatch):
        """A slice that never went away must not re-download."""
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts")
        _write(ckpt_dir / "checkpoint_500.pt")
        with mock.patch.object(utils, "_gcloud_storage") as g:
            assert utils.fetch_checkpoint_from_gcs(ckpt_dir) is None
        g.assert_not_called()

    def test_picks_highest_step(self, ckpt_dir, monkeypatch):
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts")
        run = ckpt_dir.name
        listing = "\n".join([
            f"gs://bucket/ckpts/{run}/checkpoint_500.pt",
            f"gs://bucket/ckpts/{run}/checkpoint_2000.pt",
            f"gs://bucket/ckpts/{run}/checkpoint_1000.pt",
        ])
        calls = []

        def fake(*args):
            calls.append(args)
            if args[0] == "ls":
                return True, listing, ""
            _write(ckpt_dir / "checkpoint_2000.pt")
            return True, "", ""

        with mock.patch.object(utils, "_gcloud_storage", side_effect=fake):
            got = utils.fetch_checkpoint_from_gcs(ckpt_dir)

        assert got == ckpt_dir / "checkpoint_2000.pt"
        assert calls[1] == ("cp", f"gs://bucket/ckpts/{run}/checkpoint_2000.pt",
                            str(ckpt_dir))

    def test_empty_listing_returns_none(self, ckpt_dir, monkeypatch):
        """A first run has an empty prefix; that is normal, not an error."""
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts")
        with mock.patch.object(utils, "_gcloud_storage",
                               return_value=(False, "", "One or more URLs matched no objects")):
            assert utils.fetch_checkpoint_from_gcs(ckpt_dir) is None


class TestResolveResume:
    """`TAYAVISION_RESUME` was inert: the launcher printed `resume=auto` and
    nothing read the variable, so QR metadata and SPEC.md both described a
    self-heal that could not happen."""

    @pytest.mark.parametrize("value", ["", "off", "none", "NO", "0", "False", "  "])
    def test_off_values_start_fresh(self, monkeypatch, value):
        monkeypatch.setenv("TAYAVISION_RESUME", value)
        with mock.patch.object(utils, "_gcloud_storage") as g:
            assert utils.resolve_resume_run_id() is None
        g.assert_not_called()

    def test_unset_starts_fresh(self, monkeypatch):
        monkeypatch.delenv("TAYAVISION_RESUME", raising=False)
        assert utils.resolve_resume_run_id() is None

    def test_literal_run_id_passes_through(self, monkeypatch):
        monkeypatch.setenv("TAYAVISION_RESUME", "a2d64c8e-ab98-405e-95f8-3aa7f1a1835d")
        with mock.patch.object(utils, "_gcloud_storage") as g:
            assert utils.resolve_resume_run_id() == "a2d64c8e-ab98-405e-95f8-3aa7f1a1835d"
        g.assert_not_called(), "a literal id must not need a bucket listing"

    def test_auto_without_gcs_starts_fresh(self, monkeypatch):
        """auto + node-local checkpoints has nothing durable to resume from."""
        monkeypatch.setenv("TAYAVISION_RESUME", "auto")
        monkeypatch.setenv("SAVE_CKPT_DIR", "/models")
        assert utils.resolve_resume_run_id() is None

    def test_auto_picks_newest_by_timestamp(self, monkeypatch):
        """Newest wins by mtime, NOT by step number -- a later run legitimately
        has fewer steps than an earlier long one."""
        monkeypatch.setenv("TAYAVISION_RESUME", "auto")
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts")
        listing = (
            "  69294193  2026-07-25T19:38:00Z  gs://bucket/ckpts/old-run/checkpoint_4360.pt\n"
            "  69294193  2026-07-26T01:35:37Z  gs://bucket/ckpts/new-run/checkpoint_8.pt\n"
            "TOTAL: 2 objects, 138588386 bytes (132.17MiB)\n"
        )
        with mock.patch.object(utils, "_gcloud_storage",
                               return_value=(True, listing, "")):
            assert utils.resolve_resume_run_id() == "new-run"

    def test_auto_ignores_total_summary_line(self, monkeypatch):
        """The `TOTAL:` footer has no gs:// field and must not be parsed."""
        monkeypatch.setenv("TAYAVISION_RESUME", "auto")
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts")
        listing = "TOTAL: 0 objects, 0 bytes (0B)\n"
        with mock.patch.object(utils, "_gcloud_storage",
                               return_value=(True, listing, "")):
            assert utils.resolve_resume_run_id() is None

    def test_auto_on_empty_prefix_starts_fresh(self, monkeypatch):
        monkeypatch.setenv("TAYAVISION_RESUME", "auto")
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts")
        with mock.patch.object(utils, "_gcloud_storage",
                               return_value=(False, "", "matched no objects")):
            assert utils.resolve_resume_run_id() is None


class TestSaveCheckpointIntegration:
    """`save_checkpoint` must mirror exactly what it wrote, once."""

    def _args(self, ckpt_dir):
        projector = __import__("torch").nn.Linear(2, 2)
        model = mock.Mock()
        model.multi_modal_projector = projector
        model._orig_mod = None
        del model._orig_mod  # keep _unwrap_model from following a Mock
        del model.module
        opt = mock.Mock()
        opt.state_dict.return_value = {}
        sched = mock.Mock()
        sched.state_dict.return_value = {}
        return ckpt_dir, 40, model, opt, sched

    def test_mirrors_after_local_write(self, ckpt_dir, monkeypatch):
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts")
        args = self._args(ckpt_dir)
        with mock.patch.object(utils, "mirror_checkpoint_to_gcs") as m:
            utils.save_checkpoint(*args)
        expected = ckpt_dir / "checkpoint_40.pt"
        assert expected.exists(), "local checkpoint must still be written"
        m.assert_called_once_with(expected)

    def test_only_the_writer_mirrors(self, ckpt_dir, monkeypatch):
        """On a non-writing rank nothing is uploaded -- otherwise N hosts race
        to write the same object."""
        monkeypatch.setenv("SAVE_CKPT_DIR", "gs://bucket/ckpts")
        backend = mock.Mock()
        backend.is_main = False
        args = self._args(ckpt_dir)
        with mock.patch.object(utils, "mirror_checkpoint_to_gcs") as m:
            utils.save_checkpoint(*args, backend=backend)
        backend.save.assert_called_once()
        m.assert_not_called()
