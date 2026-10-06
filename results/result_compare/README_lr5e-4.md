# Comparison trial: LR 0.0005, seed 123

Run from the `Gen_Image_data_processing` repository root in PowerShell:

```powershell
$py = 'C:\Users\35379\.venvs\genimage-cu121\Scripts\python.exe'
$data = 'F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive'

& $py train_final.py --model m2_baseline_a --seed 123 --data-root $data --protocol configs/result_compare_lr5e-4_a.json
& $py train_final.py --model pretrained_frequency_only --seed 123 --data-root $data --protocol configs/result_compare_lr5e-4_frequency.json
& $py train_final.py --model pretrained_baseline_b --seed 123 --data-root $data --protocol configs/result_compare_lr5e-4_b.json
```

This trial uses the original M2 spatial-only Baseline A, M3's pretrained frequency-only model, and M4's fused Baseline B. Compared with the LR 0.0004 trial, only the learning rate changes. Shared settings remain: seed 123, maximum 30 epochs, no early stopping in epochs 1–10, then patience 5 on validation AUROC; Adam, batch size 32, BCEWithLogitsLoss, balanced training sampler, weight decay 0.0001, gradient clipping 1.0, CUDA. The best checkpoint is selected by validation AUROC. Validation uses its official distribution; the test set is not used for selection.

Each run goes to `results/result_compare/lr_5e-4/<model>_seed123_<timestamp>/` and records the 12-column per-epoch CSV, `best.pt`, `last.pt`, `config.json`, `protocol.json`, `run_summary.json`, `data_audit.json`, and `environment.json`.
