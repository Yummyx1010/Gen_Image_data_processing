# M5 scoring

`score_predictions.py` reads completed prediction CSVs. It does not load models, open images or train networks. The current scores are in `results/m5_scores_20261008_184008_9498f1/`.

Training commit: `58f1717aeb50da25c8db7bbebe5bbd7c38a9e266`. Evaluation uses nine A/F/B checkpoints (seeds 42, 123 and 2026), trained with learning rate 0.0009. Predictions are from `results/m5_test_20261008_124810_081d31/`.

## Metric definitions

AI-generated images are the positive class (1), and Nature real images are the negative class (0). Each generator is evaluated using its 2,500 AI images and the same 1,166 real images: 3,666 images per cohort.

P(AI) >= 0.5 is classified as AI. TP and FN refer to AI images; FP and TN refer to real images.

- Accuracy = (TP + TN) / (TP + FN + FP + TN).
- Precision = TP / (TP + FP). A zero denominator remains undefined rather than being replaced with zero.
- Recall = TP / (TP + FN).
- AUROC is calculated with scikit-learn `roc_auc_score` from continuous `pred_probability`, not binary predicted labels.

For each model and seed:

- Macro AUROC is the unweighted mean of the three generator AUROCs.
- Cross-generator SD is their population standard deviation (ddof=0).
- Worst-generator AUROC is their minimum.
- AUROC range is their maximum minus minimum.

Each statistic is then summarised across seeds 42, 123 and 2026 using its mean and sample SD (ddof=1). Predictions from different seeds are not combined into an ensemble. No significance tests were performed. Full numerical precision is retained; rounding is for display only.

Accuracy, precision and recall depend on the threshold and class proportions. The cohort ratio is 2,500 AI to 1,166 real images, with no balanced test sampling. AUROC is the primary performance metric. Interpret cross-generator SD alongside mean and worst-generator AUROC: consistently low scores do not indicate a good detector. Seed SD is not a confidence interval.

## Reproduction

```sh
python -B m5/test_scoring.py
python -B m5/score_predictions.py --run-dir results/m5_test_20261008_124810_081d31
```

The scoring command creates a new results folder. Existing results do not need to be recomputed for inspection.

## Excel export

The workbook retains formulas for accuracy, precision, recall, means and SDs. XlsxWriter is required for export:

```sh
python -B m5/build_scores_workbook.py results/m5_scores_20261008_184008_9498f1/scores.json --output-dir results/workbook_export
```

Use a new output folder. The exported file is `M5_results_comparison.xlsx`, with sheets **Model comparison** and **Run metrics**. Formula caches come from `scores.json`; Excel recalculates on open.
