"""Focused tests of the M5 adapter, using artificial values, not real images."""
import csv
from contextlib import redirect_stderr
import io
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

import evaluate


class ArtificialRows(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return {"image": torch.tensor([[-2.0, 0.0, 2.0][index]]),
                "label": [0, 1, 1][index], "path": f"folder\\image{index}.jpg",
                "generator": ["Nature", "glide", "wukong"][index]}


class ArtificialModel(torch.nn.Module):
    def forward(self, values):
        if self.training or not torch.is_inference_mode_enabled():
            raise AssertionError("The adapter must disable training and autograd")
        return values


class RunnerTests(unittest.TestCase):
    def test_windows_paths_and_unsafe_paths(self):
        self.assertEqual(evaluate.relative_path(r"Nature\picture.JPEG"), "Nature/picture.JPEG")
        for path in ("", "/absolute.jpg", r"C:\image.jpg", "../image.jpg",
                     "folder/../image.jpg", "folder//image.jpg", "./image.jpg"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                evaluate.relative_path(path)

    def test_safe_default_and_invalid_limits(self):
        self.assertEqual(evaluate.parse_args([]).mode, "check")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate.parse_args(["--limit", "10"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate.parse_args(["--mode", "predict", "--limit", "0"])

    def test_order_labels_fake_probability_and_final_short_batch(self):
        loader = DataLoader(ArtificialRows(), batch_size=2, shuffle=False)
        with tempfile.TemporaryDirectory(prefix="m5_test_", dir=evaluate.PROJECT_ROOT / "logs/tmp") as directory:
            path = Path(directory) / "artificial.csv"
            result = evaluate.predict_to_csv(ArtificialModel(), loader, path,
                                            {"model": "A", "seed": 42}, "cpu", 0.5)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(result["rows"], 3)
            self.assertEqual([r["row_index"] for r in rows], ["0", "1", "2"])
            self.assertEqual([r["sample_id"] for r in rows],
                             [f"folder/image{i}.jpg" for i in range(3)])
            self.assertEqual([r["true_label"] for r in rows], ["0", "1", "1"])
            self.assertEqual([r["pred_label"] for r in rows], ["0", "1", "1"])
            for row, logit in zip(rows, [-2, 0, 2]):
                self.assertAlmostEqual(float(row["pred_probability"]),
                                       1 / (1 + math.exp(-logit)), places=6)
            self.assertFalse(path.with_suffix(".csv.part").exists())

    def test_nonfinite_outputs_never_become_complete_csv(self):
        class NonfiniteModel(torch.nn.Module):
            def forward(self, values):
                return torch.full_like(values, float("nan"))

        loader = DataLoader(ArtificialRows(), batch_size=2)
        with tempfile.TemporaryDirectory(prefix="m5_test_", dir=evaluate.PROJECT_ROOT / "logs/tmp") as directory:
            path = Path(directory) / "invalid.csv"
            with self.assertRaisesRegex(ValueError, "finite model logits"):
                evaluate.predict_to_csv(NonfiniteModel(), loader, path,
                                        {"model": "B", "seed": 123}, "cpu", 0.5)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
