# Training source hashes

The training checkpoints and `environment.json` files store SHA-256 hashes of
the exact source-file bytes read on the training machine. The original Windows
checkout used CRLF for some Python files, while Git's former `* text=auto`
rule stored those files with LF. This made the hashes of GitHub's raw files
different despite identical Python code.

`.gitattributes` now disables line-ending conversion for the training source:
`dataset.py`, `train_final.py`, and everything under `models/` and
`training/`. The existing source files are re-added to Git with their original
bytes. This source-byte fix did not change any model tensors. One later
checkpoint metadata correction is documented below.

To guard future training runs, execute:

```powershell
python -m unittest discover -s tests -p test_source_hash_checkout.py -v
```

Run this check before committing or training, and commit the source version
used for each experiment. A checkpoint's hashes identify its **training-time**
source version; a later change to model or training code is expected to have
different hashes and must not be compared with an older checkpoint as if it
were the same version.

The seed-2026 and seed-42 protocols at learning rate 0.0009 have separate
config files. The completed seed-2026 B run was trained with `config.seed=2026`
but originally carried a seed-123 *phase label*. Its `protocol.json` and the
embedded protocol metadata in `best.pt` and `last.pt` were corrected to
seed-2026. The checkpoint model tensors, optimizer state, and source hashes
were verified unchanged. `provenance_correction.json` beside that run records
the original and corrected checkpoint file hashes.
