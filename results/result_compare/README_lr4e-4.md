# Comparison trial: LR 0.0004, seed 123

From the repository root in PowerShell, use the same data root for all models:

```powershell
$py = 'C:\Users\35379\.venvs\genimage-cu121\Scripts\python.exe'
$data = 'F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive'

& $py train_final.py --model m2_baseline_a --seed 123 --data-root $data --protocol configs/result_compare_lr4e-4_a.json
& $py train_final.py --model pretrained_frequency_only --seed 123 --data-root $data --protocol configs/result_compare_lr4e-4_frequency.json
& $py train_final.py --model pretrained_baseline_b --seed 123 --data-root $data --protocol configs/result_compare_lr4e-4_b.json
```

The three models are the original M2 spatial-only Baseline A, M3's pretrained frequency-only model, and M4's spatial-frequency Baseline B. The training protocol is unchanged from the LR 0.0003 trial except for the learning rate. Shared settings are seed 123, batch size 32, Adam, maximum 30 epochs, no early stopping in epochs 1–10, then patience 5 on validation AUROC, and best checkpoint selected by validation AUROC. Training uses balanced sampling; validation retains its original distribution. Loss is BCEWithLogitsLoss, weight decay 0.0001, gradient clipping 1.0, CUDA.

Run outputs are written to `results/result_compare/lr_4e-4/<model>_seed123_<timestamp>/`. Each run writes a 12-column per-epoch training CSV and records parameters in `config.json`, `protocol.json`, and `run_summary.json`, alongside `best.pt` and `last.pt`.
