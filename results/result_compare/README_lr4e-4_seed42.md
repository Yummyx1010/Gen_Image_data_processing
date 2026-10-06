# Comparison repeat: LR 0.0004, seed 42

Run from the `Gen_Image_data_processing` repository root in PowerShell:

```powershell
$py = 'C:\Users\35379\.venvs\genimage-cu121\Scripts\python.exe'
$data = 'F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive'

& $py train_final.py --model m2_baseline_a --seed 42 --data-root $data --protocol configs/result_compare_lr4e-4_seed42_a.json
& $py train_final.py --model pretrained_frequency_only --seed 42 --data-root $data --protocol configs/result_compare_lr4e-4_seed42_frequency.json
& $py train_final.py --model pretrained_baseline_b --seed 42 --data-root $data --protocol configs/result_compare_lr4e-4_seed42_b.json
```

This repeats the LR 0.0004 comparison with seed 42. The architectures and all other training settings match the seed 123 and seed 2026 runs: original M2 Baseline A, M3 pretrained frequency-only model, and M4 fused Baseline B; Adam, batch size 32, maximum 30 epochs, no early stopping in epochs 1–10, then patience 5 on validation AUROC, best checkpoint selected by validation AUROC, balanced sampling on train only, BCEWithLogitsLoss, weight decay 0.0001, gradient clipping 1.0, and CUDA. No test-set data is used for checkpoint selection.

Each run is saved under `results/result_compare/lr_4e-4/<model>_seed42_<timestamp>/`. Its per-epoch CSV retains train/validation loss, accuracy, precision, recall, AUROC, and epoch duration. The run also writes `best.pt`, `last.pt`, `config.json`, `protocol.json`, `run_summary.json`, `data_audit.json`, and `environment.json`.
