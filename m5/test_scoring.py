"""Hand-calculated checks for statistical scoring; no images or models."""
import math
import unittest
from score_predictions import calculate_metrics, aggregate


class ScoringTests(unittest.TestCase):
    def test_known_auc_and_confusion_counts(self):
        result = calculate_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (2, 0, 1, 1))
        self.assertEqual((result["auroc"], result["accuracy"], result["precision"], result["recall"]), (0.75, 0.75, 1.0, 0.5))

    def test_ties_and_threshold_boundary(self):
        result = calculate_metrics([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual((result["auroc"], result["accuracy"], result["precision"], result["recall"]), (0.5, 0.5, 0.5, 1.0))

    def test_undefined_precision_is_not_zero(self):
        result = calculate_metrics([0, 1], [0.1, 0.4])
        self.assertIsNone(result["precision"])
        self.assertFalse(result["precision_defined"])
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["recall"], 0.0)

    def test_invalid_scores_and_missing_classes_fail(self):
        with self.assertRaises(ValueError):
            calculate_metrics([0, 1], [math.nan, 0.7])
        with self.assertRaises(ValueError):
            calculate_metrics([1, 1], [0.3, 0.7])

    def test_seed_sd_and_generator_sd_are_distinct(self):
        rows = []
        for model in ("A", "F", "B"):
            for shift, seed in enumerate((42, 123, 2026)):
                for index, generator in enumerate(("glide", "Midjourney", "wukong")):
                    rows.append({"model": model, "seed": seed, "generator": generator,
                                 "auroc": 0.6 + 0.1 * index + 0.1 * shift,
                                 "accuracy": 0.8, "precision": 0.9, "recall": 0.7})
        stability, summaries, models, differences = aggregate(rows)
        self.assertAlmostEqual(stability[0]["cross_generator_sd"], math.sqrt(0.02 / 3))
        self.assertAlmostEqual(models[0]["macro_auroc_mean"], 0.8)
        self.assertAlmostEqual(models[0]["macro_auroc_seed_sd"], 0.1)
        self.assertAlmostEqual(models[0]["worst_generator_auroc_mean"], 0.7)
        self.assertEqual(len(summaries), 9)
        self.assertTrue(all(x["B_minus_A"] == 0 for x in differences))


if __name__ == "__main__":
    unittest.main()
