# Comparison repeat: LR 0.0004, seed 2026

Run from the `Gen_Image_data_processing` repository root in PowerShell:

```powershell
$py = 'C:\Users\35379\.venvs\genimage-cu121\Scripts\python.exe'
$data = 'F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive'

& $py train_final.py --model m2_baseline_a --seed 2026 --data-root $data --protocol configs/result_compare_lr4e-4_seed2026_a.json
& $py train_final.py --model pretrained_frequency_only --seed 2026 --data-root $data --protocol configs/result_compare_lr4e-4_seed2026_frequency.json
& $py train_final.py --model pretrained_baseline_b --seed 2026 --data-root $data --protocol configs/result_compare_lr4e-4_seed2026_b.json
```

The model architectures and training protocol match the LR 0.0004, seed 123 comparison. Only the seed changes. Shared settings: Adam, learning rate 0.0004, batch size 32, maximum 30 epochs, no early stopping in epochs 1–10, then patience 5 based on validation AUROC; best checkpoint also selected by validation AUROC. Training uses balanced sampling; validation retains its original distribution. Loss is BCEWithLogitsLoss, weight decay 0.0001, gradient clipping 1.0, CUDA.

Each run is saved under `results/result_compare/lr_4e-4/<model>_seed2026_<timestamp>/`. The per-epoch CSV retains train/validation loss, accuracy, precision, recall, AUROC, and epoch duration. The run also saves `best.pt`, `last.pt`, `config.json`, `protocol.json`, `run_summary.json`, `data_audit.json`, and `environment.json`.
