# Controlled comparison: spatial, frequency, and fusion

The three current models are:

| CLI model | Architecture | Implementation |
| --- | --- | --- |
| `m2_baseline_a` | Frozen pretrained ResNet-18 spatial encoder (512-D) and `Linear(512, 1)` | `models/baseline_a.py`, `models/baseline_a_adapter.py` |
| `pretrained_frequency_only` | Trainable pretrained ResNet-18 on the normalized FFT log magnitude spectrum (256-D projection) and `Linear(256, 1)` | `models/frequency_encoder_pretrained.py`, `models/frequency_only.py` |
| `baseline_b` | Frozen pretrained ResNet-18 spatial encoder (512-D) plus the original trainable lightweight frequency CNN (256-D), concatenated into 768-D and classified by `Linear(768, 1)` | `models/spatial_encoder.py`, `models/frequency_encoder.py`, `models/baseline_b.py` |

The current B follows the presentation architecture literally: `spatial = SpatialEncoder(image)` and `frequency = FrequencyEncoder(image)`; then `torch.cat((spatial, frequency), dim=1)` produces a real 768-D tensor passed to one `Linear(768, 1)` head. A remains unchanged. Both A and B use a linear binary head, but their head dimensions and randomly initialized weights differ. The frequency CNN is trainable; the spatial ResNet-18 is frozen. B therefore has more trainable parameters and compute than A, so any performance difference cannot be attributed to frequency information alone.

`frequency_cnn_concat_v1` is the current B architecture identifier and new run directories contain `_cnn_concat_v1_`. Historical `pretrained_fusion_v1` and `pretrained_fusion_linear_v2` checkpoints remain loadable under the legacy `pretrained_baseline_b` name. They use a different frequency encoder and must not be reported as results of the current B. Existing results are preserved, but current B must be trained from scratch. The separate `pretrained_frequency_only` model is also unchanged and uses a pretrained ResNet-18 rather than B's lightweight frequency CNN.

## Run the three models

Run from the repository root with the same image root, seed, and paired protocol settings. For example, the existing learning rate 0.0004, seed 123 configurations are:

```powershell
$py = 'C:\Users\35379\.venvs\genimage-cu121\Scripts\python.exe'
$dataRoot = 'F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive'

& $py train_final.py --model m2_baseline_a --seed 123 --data-root $dataRoot --protocol configs/result_compare_lr4e-4_a.json
& $py train_final.py --model pretrained_frequency_only --seed 123 --data-root $dataRoot --protocol configs/result_compare_lr4e-4_frequency.json
& $py train_final.py --model baseline_b --seed 123 --data-root $dataRoot --protocol configs/result_compare_lr4e-4_b.json
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
