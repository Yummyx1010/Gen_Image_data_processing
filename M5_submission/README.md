# M5 Evaluation and Analysis

The original A, F and B models were evaluated with seeds 42, 123 and 2026. All nine checkpoints use training learning rate 0.0009. Each predicts the complete 8,666-image test set, producing 77,994 records.

## Start here

- [Results and conclusion](logs/M5_metrics_summary.md)
- [Excel comparison](results/m5_scores_20261008_184008_9498f1/M5_results_comparison.xlsx)
- [Figure descriptions](results/m5_figures_original_20261008_232856_d8c2bd/FIGURES.md)
- [Error analysis](results/m5_error_examples_original_c776b30a/ERROR_ANALYSIS.md)
- [Case gallery](results/m5_error_examples_original_c776b30a/04_error_examples.png)

## Models and test protocol

| Model | Features | Classification head |
|---|---|---|
| A | Frozen spatial ResNet-18 | Linear(512, 1) |
| F | M3 pretrained frequency encoder | Linear(256, 1) |
| B | Spatial and frequency features concatenated | Linear(768, 1) |

F and B use the same frequency-encoder implementation with separately trained weights.

The test set contains 1,166 Nature real images and 2,500 images each from glide, Midjourney and wukong. Real=0 and AI=1. The original run used CPU, batch size 32, no shuffling and threshold 0.5. AUROC uses continuous P(AI).

## Contents

| Folder | Contents |
|---|---|
| m5/ | Evaluation, scoring, plotting and error-analysis code |
| results/ | Full predictions, metric tables, workbook, figures and analysis |
| logs/ | Final conclusion and input manifests |
| code/ | Original test CSV |
| data/ | Six original images used in the error analysis |

The data folder contains only the six case images. The prediction CSVs and test manifest cover all 8,666 images.

## Recalculate metrics

From this folder, with Python 3.10 or later and scikit-learn installed:

```sh
python -B m5/test_scoring.py
python -B m5/score_predictions.py --run-dir results/m5_test_20261008_124810_081d31
```

The scoring command creates a new results folder. [SCORING.md](m5/SCORING.md) gives the metric definitions and Excel export command. Workbook export requires XlsxWriter. Plot generation uses base R at the Rscript path in the script. Gallery generation requires Pillow and the specified Arial font paths.

## Repeat image inference

Use the original project workspace with the frozen M4 source code, nine checkpoints and complete image dataset. [model_index.json](logs/model_index.json) records checkpoint paths and hashes at training commit 58f1717aeb50da25c8db7bbebe5bbd7c38a9e266.

```sh
python -B m5/evaluate.py --mode check
python -B m5/evaluate.py --mode predict
```

Inference requires PyTorch and torchvision. The original-workspace verification script also uses the saved diagnostic run. Historical run records retain execution-time paths and source hashes.
