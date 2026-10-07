# Controlled comparison: spatial, frequency, and fusion

The three current models are:

| CLI model | Architecture | Implementation |
| --- | --- | --- |
| `m2_baseline_a` | Frozen pretrained ResNet-18 spatial encoder (512-D) and `Linear(512, 1)` | `models/baseline_a.py`, `models/baseline_a_adapter.py` |
| `pretrained_frequency_only` | Trainable pretrained ResNet-18 on the normalized FFT log magnitude spectrum (256-D projection) and `Linear(256, 1)` | `models/frequency_encoder_pretrained.py`, `models/frequency_only.py` |
| `baseline_b` | Frozen A spatial ResNet-18 (512-D) plus the **trainable** M3 F frequency encoder architecture (256-D), concatenated into 768-D and classified by `Linear(768, 1)` | `models/spatial_encoder.py`, `models/frequency_encoder_pretrained.py`, `models/baseline_b.py` |

The current B constructs the same `PretrainedFrequencyEncoder` class used by F. Its ResNet-18 backbone starts from ImageNet weights, its 512-to-256 projection starts from a new random initialization, and both are updated during B training. **No trained F `best.pt` weights are loaded.** A's spatial encoder is unchanged and frozen. A, F and B use one linear binary classifier each, with input dimensions 512, 256 and 768 respectively; classifier weights are not shared. B concatenates actual `[B,512]` and `[B,256]` feature tensors before its classifier.

`m3_frequency_concat_v1` is the current B architecture identifier. Existing `frequency_cnn_concat_v1` runs are historical **CNN-B** results, not results of this A+M3-F architecture. Their saved checkpoints remain loadable, but the top-level B protocol files now describe the new model. Historical `pretrained_fusion_v1` and `pretrained_fusion_linear_v2` checkpoints also remain loadable under `pretrained_baseline_b`. Do not pool or relabel results across these architectures. New B requires a new training run.

B and F are independently trained from ImageNet-initialized ResNet-18 backbones; B does not require an F training run first. Even with matched CSVs, seed, optimizer, and settings, B versus A is **not** a strict single-factor causal test of frequency information: B adds a trainable encoder, projection, parameters and computation. Reserve the test split for final evaluation.

## Run the three models

Run from the repository root with the same image root, seed, and paired protocol settings. F and B can be trained independently; no F checkpoint path is needed. For example, learning rate 0.0004 and seed 123:

```powershell
$py = 'C:\Users\35379\.venvs\genimage-cu121\Scripts\python.exe'
$dataRoot = 'F:\AAA-nz-auckland-homework\760\Group 12-Milestone 1\archive'

& $py train_final.py --model m2_baseline_a --seed 123 --data-root $dataRoot --protocol configs/result_compare_lr4e-4_a.json
& $py train_final.py --model pretrained_frequency_only --seed 123 --data-root $dataRoot --protocol configs/result_compare_lr4e-4_frequency.json
& $py train_final.py --model baseline_b --seed 123 --data-root $dataRoot --protocol configs/result_compare_lr4e-4_b.json
```

Use the matching A/F/B protocol triplet for another learning rate or seed. The saved `protocol.json`, `data_audit.json`, and `environment.json` identify settings, CSV hashes and source hashes. Previously trained CNN-B results remain historical; they cannot represent the updated B. If exact training-entry source identity is required across all three models, rerun A and F as well because `train_final.py` changed, although their model implementations and training branches were not altered.

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
| Frequency backbone frozen inside B | `false` for the current A+M3-F experiment | `false`; frequency backbone, projection and B's linear head train |

Prefer a small, declared grid and select hyperparameters using validation AUROC only. The validation fake class is VQDM; repeated broad tuning on that single generator can overfit the model-selection process. Keep `test.csv` for one final evaluation after choosing the protocol. Describe B as an ImageNet-initialized trainable frequency fusion experiment, not as loading a trained F checkpoint.
