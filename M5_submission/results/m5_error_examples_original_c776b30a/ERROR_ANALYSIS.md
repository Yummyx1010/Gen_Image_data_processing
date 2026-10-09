# Error analysis

A is spatial-only, F is frequency-only and B combines spatial and frequency features. Results use seeds 42, 123 and 2026 on 8,666 test images.

## Case selection

The six images were selected by source coverage and filename order, with two images per A/B error pattern shared by all three seeds.

P(AI) >= 0.5 is classified as AI. The gallery reports correct classifications out of three seeds.

## Visual observations

| Case | Source / true class | Visible content | A correct | F correct | B correct |
|---|---|---|---:|---:|---:|
| 1 | glide / AI-generated image | Coral-like structures appear against a blue-green background, with dense curved textures in orange-yellow areas. | 3/3 | 0/3 | 0/3 |
| 2 | Midjourney / AI-generated image | A close-up of a seashell on sand. The foreground is relatively sharp, while the sea and sky are blurred. | 3/3 | 0/3 | 0/3 |
| 3 | wukong / AI-generated image | Two brightly coloured birds perch on a branch against a mostly blurred green background. | 0/3 | 0/3 | 3/3 |
| 4 | Nature / Real image | White curtains surround a wooden bed. The area behind the bed is brightly lit and the foreground is darker. | 0/3 | 2/3 | 3/3 |
| 5 | glide / AI-generated image | A small grey-white bird perches among thin branches. The background has bright areas and the image is generally blurred. | 0/3 | 2/3 | 0/3 |
| 6 | Midjourney / AI-generated image | A close-up of a yellow fish head with a clearly visible eye and mouth against a blurred dark-green background. | 0/3 | 0/3 | 0/3 |

## Findings

In cases 1 and 2, A detects the AI images while B classifies them as real in all three seeds. In cases 3 and 4, B correctly classifies an AI bird image and a real bedroom image that A misclassifies. Both A and B classify the AI images in cases 5 and 6 as real.

In case 5, F is correct in two seeds while B is wrong in all three. F and B are trained separately; B uses frequency features rather than F's final classification.

## Full-test results

Mean AUROC is 0.9515 for A and 0.9411 for B; cross-generator SD is 0.0166 for A and 0.0280 for B. The frequency branch did not improve mean AUROC or cross-generator consistency under the evaluated settings.

Mean AUROC and cross-generator SD are calculated within each seed and then averaged across seeds.

## Supporting data

Sample IDs and all 54 per-seed probabilities and predictions are in CASE_PREDICTIONS.md and case_details.json.
