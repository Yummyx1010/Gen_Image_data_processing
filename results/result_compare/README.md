# Result compare: learning rate 0.0003, seed 123

Run from the repository root with the same `--data-root` for all three models:

```powershell
python train_final.py --model m2_baseline_a --seed 123 --data-root "F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive" --protocol configs/result_compare_a.json
python train_final.py --model pretrained_frequency_only --seed 123 --data-root "F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive" --protocol configs/result_compare_frequency.json
python train_final.py --model pretrained_baseline_b --seed 123 --data-root "F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive" --protocol configs/result_compare_b.json
```

| Model argument | Architecture and source |
| --- | --- |
| `m2_baseline_a` | Original M2 spatial-only Baseline A: `models/baseline_a.py` via `models/baseline_a_adapter.py` |
| `pretrained_frequency_only` | M3 pretrained frequency encoder: `models/frequency_encoder_pretrained.py`, with the classifier in `models/frequency_only.py` |
| `pretrained_baseline_b` | M4 fusion model: `models/pretrained_baseline_b.py`, using M2 spatial and M3 frequency features |

Shared protocol: Adam, learning rate 0.0003, seed 123, batch size 32, maximum 30 epochs, no early stopping in epochs 1–10, then patience 5; best checkpoint and early stopping use validation AUROC. Training uses `WeightedRandomSampler`; validation retains its original distribution. Loss is `BCEWithLogitsLoss`, weight decay 0.0001 and gradient clipping 1.0. No test-set evaluation occurs in these runs.

Each run is saved under `results/result_compare/lr_3e-4/<model>_seed123_<timestamp>/`. The per-epoch `training_log_<model>_seed123.csv` retains train/validation loss, accuracy, precision, recall, AUROC, and epoch duration. Each run also writes `best.pt`, `last.pt`, `config.json`, `protocol.json`, `run_summary.json`, `data_audit.json`, and `environment.json`.
