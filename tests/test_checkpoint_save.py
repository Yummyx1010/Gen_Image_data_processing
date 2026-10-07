"""Checkpoint writes survive Windows refusing to replace an existing file."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from training.train import save_checkpoint


class CheckpointSaveTests(unittest.TestCase):
    def test_transient_lock_retries_the_completed_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "last.pt"
            torch.save({"epoch": 1}, target)
            original_replace = Path.replace
            attempts = 0

            def replace(source, destination):
                nonlocal attempts
                if Path(destination) == target:
                    attempts += 1
                    if attempts <= 2:
                        raise PermissionError(5, "Access is denied")
                return original_replace(source, destination)

            with patch.object(Path, "replace", replace), patch("training.train.time.sleep"):
                saved = save_checkpoint(target, {"epoch": 2}, allow_fallback=True)

            self.assertEqual(attempts, 3)
            self.assertEqual(saved, target)
            self.assertEqual(torch.load(target, weights_only=True)["epoch"], 2)

    def test_persistent_target_lock_keeps_a_distinct_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "best.pt"
            torch.save({"epoch": 1}, target)
            original_replace = Path.replace

            def replace(source, destination):
                if Path(destination) == target:
                    raise PermissionError(5, "Access is denied")
                return original_replace(source, destination)

            with patch.object(Path, "replace", replace), patch("training.train.time.sleep"):
                saved = save_checkpoint(target, {"epoch": 2}, allow_fallback=True)

            self.assertNotEqual(saved, target)
            self.assertTrue(saved.exists())
            self.assertTrue(saved.name.startswith("best_epoch2_"))
            self.assertEqual(torch.load(target, weights_only=True)["epoch"], 1)
            self.assertEqual(torch.load(saved, weights_only=True)["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
