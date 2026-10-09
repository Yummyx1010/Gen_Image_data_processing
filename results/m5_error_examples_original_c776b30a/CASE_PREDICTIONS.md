# Per-seed predictions for the six cases

Probabilities are P(AI), with a fixed threshold of 0.5. Ground truth comes from the original test manifest. Probabilities and model predictions are reported separately.

## Case 1: glide/GLIDE_1000_200_01_109_glide_00035.jpg

Original image: [view image](<../../data/glide/GLIDE_1000_200_01_109_glide_00035.jpg>). True class: AI-generated image.

| Seed | A: P(AI) / prediction | F: P(AI) / prediction | B: P(AI) / prediction |
|---|---|---|---|
| 42 | 0.6442 / AI (correct) | 0.1185 / Real (wrong) | 0.0063 / Real (wrong) |
| 123 | 0.6374 / AI (correct) | 0.0972 / Real (wrong) | 0.1671 / Real (wrong) |
| 2026 | 0.5435 / AI (correct) | 0.1962 / Real (wrong) | 0.2186 / Real (wrong) |

## Case 2: Midjourney/112_midjourney_23.jpg

Original image: [view image](<../../data/Midjourney/112_midjourney_23.jpg>). True class: AI-generated image.

| Seed | A: P(AI) / prediction | F: P(AI) / prediction | B: P(AI) / prediction |
|---|---|---|---|
| 42 | 0.5096 / AI (correct) | 0.1406 / Real (wrong) | 0.1162 / Real (wrong) |
| 123 | 0.6120 / AI (correct) | 0.0371 / Real (wrong) | 0.3232 / Real (wrong) |
| 2026 | 0.6125 / AI (correct) | 0.1290 / Real (wrong) | 0.2885 / Real (wrong) |

## Case 3: wukong/10_wukong_image72.jpg

Original image: [view image](<../../data/wukong/10_wukong_image72.jpg>). True class: AI-generated image.

| Seed | A: P(AI) / prediction | F: P(AI) / prediction | B: P(AI) / prediction |
|---|---|---|---|
| 42 | 0.3525 / Real (wrong) | 0.0578 / Real (wrong) | 0.6108 / AI (correct) |
| 123 | 0.2765 / Real (wrong) | 0.1253 / Real (wrong) | 0.7395 / AI (correct) |
| 2026 | 0.3088 / Real (wrong) | 0.2611 / Real (wrong) | 0.5217 / AI (correct) |

## Case 4: Nature/ILSVRC2012_val_00004645.JPEG

Original image: [view image](<../../data/Nature/ILSVRC2012_val_00004645.JPEG>). True class: Real image.

| Seed | A: P(AI) / prediction | F: P(AI) / prediction | B: P(AI) / prediction |
|---|---|---|---|
| 42 | 0.5822 / AI (wrong) | 0.5135 / AI (wrong) | 0.0220 / Real (correct) |
| 123 | 0.5876 / AI (wrong) | 0.0755 / Real (correct) | 0.4879 / Real (correct) |
| 2026 | 0.6065 / AI (wrong) | 0.1119 / Real (correct) | 0.3739 / Real (correct) |

## Case 5: glide/GLIDE_1000_200_00_013_glide_00106.jpg

Original image: [view image](<../../data/glide/GLIDE_1000_200_00_013_glide_00106.jpg>). True class: AI-generated image.

| Seed | A: P(AI) / prediction | F: P(AI) / prediction | B: P(AI) / prediction |
|---|---|---|---|
| 42 | 0.1357 / Real (wrong) | 0.7729 / AI (correct) | 0.0124 / Real (wrong) |
| 123 | 0.0335 / Real (wrong) | 0.2034 / Real (wrong) | 0.1981 / Real (wrong) |
| 2026 | 0.0899 / Real (wrong) | 0.7682 / AI (correct) | 0.0418 / Real (wrong) |

## Case 6: Midjourney/0_midjourney_161.jpg

Original image: [view image](<../../data/Midjourney/0_midjourney_161.jpg>). True class: AI-generated image.

| Seed | A: P(AI) / prediction | F: P(AI) / prediction | B: P(AI) / prediction |
|---|---|---|---|
| 42 | 0.1185 / Real (wrong) | 0.2794 / Real (wrong) | 0.0174 / Real (wrong) |
| 123 | 0.0555 / Real (wrong) | 0.1324 / Real (wrong) | 0.0868 / Real (wrong) |
| 2026 | 0.1038 / Real (wrong) | 0.2809 / Real (wrong) | 0.2169 / Real (wrong) |
