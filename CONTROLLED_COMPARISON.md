# Controlled comparison: spatial, frequency, and fusion

The three current models are:

| CLI model | Architecture | Implementation |
| --- | --- | --- |
| `m2_baseline_a` | Frozen pretrained ResNet-18 spatial encoder (512-D) and `Linear(512, 1)` | `models/baseline_a.py`, `models/baseline_a_adapter.py` |
| `pretrained_frequency_only` | Trainable pretrained ResNet-18 on the normalized FFT log magnitude spectrum (256-D projection) and `Linear(256, 1)` | `models/frequency_encoder_pretrained.py`, `models/frequency_only.py` |
| `pretrained_baseline_b` | An identically initialized frozen spatial encoder and linear spatial classifier, plus the frequency encoder and a bias-free `Linear(256, 1)` contribution | `models/pretrained_baseline_b.py` |

The new B computes `logit_B = Linear_A(spatial) + Linear_without_bias(frequency)`. Its spatial classifier is initialized before the frequency encoder. Given the same seed, its initial spatial encoder and classifier tensors equal A's; setting the frequency contribution to zero makes their predictions identical. B has no LayerNorm, GELU, hidden classifier layer, or Dropout. The frequency branch remains trainable, so B necessarily has more trainable parameters and compute than A. This experiment isolates the addition of that branch and its learned contribution, not parameter count.

`pretrained_fusion_linear_v2` is the new B architecture identifier. Historical `pretrained_fusion_v1` checkpoints use the old nonlinear fusion model, remain loadable, and must not be reported as results of the new B. New B run directories contain `_linear_v2_` in their names. Previously saved results require no migration, but the new B must be trained from scratch.

## Run the three models

Run from the repository root with the same image root, seed, and paired protocol settings. For example, the existing learning rate 0.0004, seed 123 configurations are:

```powershell
$py = 'C:\Users\35379\.venvs\genimage-cu121\Scripts\python.exe'
$dataRoot = 'F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive'

& $py train_final.py --model m2_baseline_a --seed 123 --data-root $dataRoot --protocol configs/result_compare_lr4e-4_a.json
& $py train_final.py --model pretrained_frequency_only --seed 123 --data-root $dataRoot --protocol configs/result_compare_lr4e-4_frequency.json
& $py train_final.py --model pretrained_baseline_b --seed 123 --data-root $dataRoot --protocol configs/result_compare_lr4e-4_b.json
```

Use seed 42 or 2026 with the corresponding `configs/result_compare_lr4e-4_seed42_*.json` or `configs/result_compare_lr4e-4_seed2026_*.json` files. The seed is supplied on the command line. The saved `protocol.json`, `data_audit.json`, and `environment.json` identify the settings, CSV hashes, and source hashes used for each run.

## Practical tuning range

For the primary comparison, change each paired A, frequency-only, and B protocol together. Keep the same training data, sampler, optimizer, batch size, selection metric, and seed. In every JSON, set `learning_rate_candidates` to include the selected `learning_rate`.

| Parameter | Suggested values | Current comparison setting |
| --- | --- | --- |
| Adam learning rate | `1e-4` to `5e-4`; try `1e-4`, `2e-4`, `3e-4` first | `3e-4`, `4e-4`, or `5e-4` by trial |
| Weight decay | `0`, `1e-4`, `5e-4` | `1e-4` |
| Batch size | `16` or `32` on a 6 GB GPU; keep equal across models | `32` |
| Maximum epochs | `20` to `40` | `30` |
| Early stopping warmup / patience | `5` to `10` / `5` to `8` epochs | `10` / `5` |
| Gradient clip norm | `0.5` to `2.0` | `1.0` |
| Random seeds | At least `42`, `123`, `2026` for the selected protocol | Three seeds at LR `4e-4` |
| Frequency backbone frozen | `false` in the main experiment; `true` only as a separate ablation | `false` |

Prefer a small, declared grid and select hyperparameters using validation AUROC only. The validation fake class is VQDM; repeated broad tuning on that single generator can overfit the model-selection process. Keep `test.csv` for one final evaluation after choosing the protocol. If B or the frequency-only model is unstable at a shared learning rate, record a separate optimization study; do not mix it into the single-variable comparison.
