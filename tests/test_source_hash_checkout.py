"""Training source bytes must survive a Git commit and checkout unchanged."""

from pathlib import Path
import subprocess
import unittest

from training.train import source_hashes


ROOT = Path(__file__).resolve().parents[1]


class SourceHashCheckoutTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / ".git").exists(), "requires a Git checkout")
    def test_git_does_not_rewrite_training_source_bytes(self):
        sources = [ROOT / relative for relative in source_hashes()]
        sources.append(ROOT / "train_final.py")
        for source in sources:
            relative = source.relative_to(ROOT).as_posix()
            with self.subTest(source=relative):
                attribute = subprocess.check_output(
                    ["git", "check-attr", "text", "--", relative],
                    cwd=ROOT,
                    text=True,
                ).strip()
                self.assertTrue(
                    attribute.endswith(": unset"),
                    f"{relative} must disable Git text conversion: {attribute}",
                )
                converted = subprocess.check_output(
                    ["git", "hash-object", f"--path={relative}", relative],
                    cwd=ROOT,
                    text=True,
                ).strip()
                original = subprocess.check_output(
                    ["git", "hash-object", "--no-filters", relative],
                    cwd=ROOT,
                    text=True,
                ).strip()
                self.assertEqual(
                    converted,
                    original,
                    f"Git would change the source bytes recorded in checkpoints: {relative}",
                )


if __name__ == "__main__":
    unittest.main()
