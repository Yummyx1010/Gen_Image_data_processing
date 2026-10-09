# Original full-test figures

The figures use the original A, F and B checkpoints with seeds 42, 123 and 2026: nine checkpoints in total. Every checkpoint was evaluated on all 8,666 original test images.

1. **01_AUROC_by_generator.png** compares AUROC on the three unseen generators. Small points show individual seeds; diamonds and error bars show the seed mean and sample SD.
2. **02_mean_and_consistency.png** shows mean AUROC (higher is better) and cross-generator SD (lower is more consistent). Both statistics are calculated within each seed, then summarised across seeds. Error bars on the SD panel show variation between training seeds.
3. **03_ROC_by_generator.png** shows ROC curves for each unseen generator. False positive rate is the proportion of real images classified as AI; true positive rate is the proportion of AI images correctly detected. Thin lines show individual seeds and thick lines show interpolated mean curves. Legend AUROCs are means of exact per-seed AUROCs, not approximate areas under the thick curves.

Each generator evaluation uses 2,500 AI images and the same 1,166 real images. Seed error bars are not confidence intervals; no significance test was performed.
