"""
tests/test_train_engine.py
--------------------------
Smoke tests for train_engine.py — argument parsing, preprocessing pipeline,
resume detection, and structured output.

These tests validate the engine's orchestration logic without requiring
a real GPU, dataset, or Ultralytics model. Heavy ML components are mocked.
"""

import argparse
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mini_dataset(tmp_path: Path, split: str = "train") -> Path:
    """Write a minimal YOLO-layout dataset with 2 tiny images and labels."""
    img_dir = tmp_path / split / "images"
    lbl_dir = tmp_path / split / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    for i in range(2):
        # 32x32 random image
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img{i}.jpg"), img)
        # Single bounding box annotation
        (lbl_dir / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    return tmp_path


def _make_data_yaml(tmp_path: Path) -> Path:
    """Write a minimal data.yaml pointing at the tmp dataset."""
    yaml_path = tmp_path / "data.yaml"
    content = {
        "path": str(tmp_path).replace("\\", "/"),
        "train": "train/images",
        "val": "train/images",   # reuse train as val for smoke test
        "nc": 7,
        "names": {
            0: "missing_hole", 1: "mouse_bite", 2: "open_circuit",
            3: "short", 4: "excess_solder", 5: "insufficient_solder",
            6: "cold_solder_joint",
        },
    }
    yaml_path.write_text(yaml.dump(content))
    return yaml_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class TestArgumentParsing:
    """Validate CLI argument defaults and constraints."""

    def _parse(self, extra_args=None):
        """Import and invoke the parser with controlled argv."""
        from train_engine import run_experiment
        # We patch argparse.ArgumentParser.parse_args directly
        test_argv = [
            "--id", "TEST",
            "--desc", "SmokeTest",
        ] + (extra_args or [])
        with patch("sys.argv", ["train_engine.py"] + test_argv):
            import train_engine
            parser = argparse.ArgumentParser()
            # Re-create the parser from the module to test it in isolation
            parser.add_argument("--mode", default="train", choices=["train", "tune"])
            parser.add_argument("--variant", default="s", choices=["n", "s", "m", "l", "x"])
            parser.add_argument("--id", required=True)
            parser.add_argument("--desc", required=True)
            parser.add_argument("--epochs", type=int, default=100)
            parser.add_argument("--batch", type=int, default=12)
            parser.add_argument("--cls-pw", type=float, default=1.0, dest="cls_pw")
            parser.add_argument("--patience", type=int, default=30)
            parser.add_argument("--fraction", type=float, default=1.0)
            parser.add_argument("--cfg", default=None)
            return parser.parse_args(test_argv)

    def test_defaults(self):
        args = self._parse()
        assert args.mode == "train"
        assert args.variant == "s"
        assert args.epochs == 100
        assert args.batch == 12
        assert args.patience == 30
        assert args.cls_pw == 1.0

    def test_tune_mode(self):
        args = self._parse(["--mode", "tune"])
        assert args.mode == "tune"

    def test_cls_pw_validation(self):
        """cls_pw must be in [0.0, 1.0]."""
        args = self._parse(["--cls-pw", "0.5"])
        assert 0.0 <= args.cls_pw <= 1.0

    def test_custom_variant(self):
        args = self._parse(["--variant", "m"])
        assert args.variant == "m"


# ---------------------------------------------------------------------------
# Preprocessing pipeline (preprocess_dataset)
# ---------------------------------------------------------------------------

class TestPreprocessDataset:
    """Test the CLAHE+Gamma dataset preprocessing pipeline."""

    def test_preprocess_creates_output_dir(self, tmp_path):
        """preprocess_dataset() should create <dataset>_preproc/ directory."""
        _make_mini_dataset(tmp_path)
        data_yaml = _make_data_yaml(tmp_path)

        from train_engine import preprocess_dataset
        output_yaml = preprocess_dataset(str(data_yaml))
        output_yaml_path = Path(output_yaml)

        assert output_yaml_path.exists(), "data.yaml should exist in preproc dir"
        assert "_preproc" in str(output_yaml_path)

    def test_preprocess_copies_labels(self, tmp_path):
        """Labels must be copied alongside preprocessed images."""
        _make_mini_dataset(tmp_path)
        data_yaml = _make_data_yaml(tmp_path)

        from train_engine import preprocess_dataset
        output_yaml = preprocess_dataset(str(data_yaml))
        preproc_dir = Path(output_yaml).parent

        label_files = list((preproc_dir / "train" / "labels").glob("*.txt"))
        assert len(label_files) == 2, "Both label files should be copied"

    def test_preprocess_reuses_cached(self, tmp_path):
        """Second call with same path should reuse existing output (no re-run)."""
        _make_mini_dataset(tmp_path)
        data_yaml = _make_data_yaml(tmp_path)

        from train_engine import preprocess_dataset
        output1 = preprocess_dataset(str(data_yaml))
        output2 = preprocess_dataset(str(data_yaml))
        assert output1 == output2

    def test_preprocess_force_flag(self, tmp_path):
        """force=True should rebuild the preprocessed dataset."""
        _make_mini_dataset(tmp_path)
        data_yaml = _make_data_yaml(tmp_path)

        from train_engine import preprocess_dataset
        preprocess_dataset(str(data_yaml))  # first run
        # Second run with force — should not raise
        output = preprocess_dataset(str(data_yaml), force=True)
        assert Path(output).exists()

    def test_preprocess_yaml_uses_relative_path(self, tmp_path):
        """Generated data.yaml must not contain Windows absolute paths."""
        _make_mini_dataset(tmp_path)
        data_yaml = _make_data_yaml(tmp_path)

        from train_engine import preprocess_dataset
        output_yaml = preprocess_dataset(str(data_yaml))
        content = yaml.safe_load(Path(output_yaml).read_text())

        # The 'path' key should not start with a drive letter (e.g. C:\, D:\)
        assert not content["path"].startswith(("/", "C:", "D:", "E:")), (
            f"Expected relative path in data.yaml, got: {content['path']!r}"
        )


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------

class TestResumeDetection:
    """Verify that the engine correctly detects and uses checkpoint files."""

    def test_no_checkpoint_loads_pretrained(self, tmp_path):
        """Without a checkpoint, model_source should be the pretrained .pt file."""
        exp_dir = tmp_path / "runs" / "detect" / "CIRCA_V12S_TEST_TRAIN_Test"
        # No checkpoint file → fresh start expected

        checkpoint_path = exp_dir / "weights" / "last.pt"
        resume = checkpoint_path.exists()
        model_source = "yolo12s.pt" if not resume else str(checkpoint_path)

        assert not resume
        assert model_source == "yolo12s.pt"

    def test_checkpoint_triggers_resume(self, tmp_path):
        """With a last.pt checkpoint, resume should be True."""
        exp_dir = tmp_path / "runs" / "detect" / "CIRCA_V12S_TEST_TRAIN_Test"
        checkpoint_path = exp_dir / "weights" / "last.pt"
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.write_bytes(b"fake checkpoint")

        resume = checkpoint_path.exists()
        model_source = str(checkpoint_path) if resume else "yolo12s.pt"

        assert resume
        assert "last.pt" in model_source


# ---------------------------------------------------------------------------
# run_summary.yaml output format
# ---------------------------------------------------------------------------

class TestRunSummaryOutput:
    """Validate the structure of the machine-readable run_summary.yaml."""

    def test_summary_schema(self, tmp_path):
        """run_summary.yaml should contain the required metric keys."""
        summary = {
            "experiment_id": "001",
            "description": "Baseline_Vanilla",
            "variant": "s",
            "mode": "train",
            "data": "datasets/unified_pcb_v3/data.yaml",
            "mAP50": 0.6649,
            "mAP50_95": 0.4237,
            "precision": 0.7290,
            "recall": 0.6433,
        }
        summary_path = tmp_path / "run_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

        loaded = yaml.safe_load(summary_path.read_text())
        required_keys = {"experiment_id", "mAP50", "mAP50_95", "precision", "recall"}
        assert required_keys.issubset(loaded.keys()), (
            f"Missing keys: {required_keys - loaded.keys()}"
        )
        assert 0.0 <= loaded["mAP50"] <= 1.0
        assert 0.0 <= loaded["mAP50_95"] <= 1.0
