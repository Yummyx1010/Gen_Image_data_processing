# Error counts on the original full test set

The original A, F and B models each use seeds 42, 123 and 2026: nine checkpoints in total. Every checkpoint was evaluated on all 8,666 original test images.

An image is classified as AI when P(AI) >= 0.5. These are fixed-threshold classification counts, not AUROC. Fewer errors do not necessarily imply higher AUROC.

| Seed | A and B correct | A correct, B wrong | A wrong, B correct | A and B wrong | A errors | F errors | B errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 5408 | 2000 | 158 | 1100 | 1258 | 3245 | 3100 |
| 123 | 6638 | 425 | 651 | 952 | 1603 | 4375 | 1377 |
| 2026 | 6912 | 546 | 348 | 860 | 1208 | 4047 | 1406 |

The first four counts in each row sum to 8,666. All three seeds are reported separately.

## Errors by image source

| Seed | Source | Images | A errors | F errors | B errors |
|---|---|---:|---:|---:|---:|
| 42 | glide | 2500 | 180 | 461 | 504 |
| 42 | Midjourney | 2500 | 536 | 1622 | 1467 |
| 42 | wukong | 2500 | 450 | 1057 | 1096 |
| 42 | Nature | 1166 | 92 | 105 | 33 |
| 123 | glide | 2500 | 273 | 1064 | 126 |
| 123 | Midjourney | 2500 | 682 | 1919 | 703 |
| 123 | wukong | 2500 | 586 | 1336 | 435 |
| 123 | Nature | 1166 | 62 | 56 | 113 |
| 2026 | glide | 2500 | 228 | 669 | 207 |
| 2026 | Midjourney | 2500 | 441 | 1761 | 686 |
| 2026 | wukong | 2500 | 431 | 1428 | 429 |
| 2026 | Nature | 1166 | 108 | 189 | 84 |

Nature contains real images; the other three sources contain AI-generated images. Each real image is counted once here. Generator-specific AUROC evaluations reuse the same real images for each generator.

## A/B patterns shared by all three seeds

| Pattern | glide | Midjourney | wukong | Nature | Total |
|---|---:|---:|---:|---:|---:|
| A and B correct | 1941 | 991 | 1292 | 994 | 5218 |
| A correct, B wrong | 11 | 78 | 35 | 6 | 130 |
| A wrong, B correct | 10 | 15 | 39 | 12 | 76 |
| A and B wrong | 79 | 308 | 222 | 18 | 627 |
| Pattern differs across seeds | 459 | 1108 | 912 | 136 | 2615 |

## Selected cases

Six cases were selected for the gallery. The selection method and analysis are in [ERROR_ANALYSIS.md](../m5_error_examples_original_c776b30a/ERROR_ANALYSIS.md); sample IDs and predictions are in [CASE_PREDICTIONS.md](../m5_error_examples_original_c776b30a/CASE_PREDICTIONS.md).
