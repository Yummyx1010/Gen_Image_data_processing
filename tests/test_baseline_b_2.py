"""Offline checks for the independent, fully frozen Baseline B_2 variant."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torchvision.models import resnet18

from models.baseline_b import BaselineB
from models.baseline_b_2 import BaselineB2
from models.frequency_encoder_frozen import FrozenFrequencyEncoder
from models.frequency_encoder_pretrained import PretrainedFrequencyEncoder
from train_baseline_b_2 import ARCHITECTURE, load_protocol, run_epoch_frozen


class BaselineB2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def models():
        factory = lambda **kwargs: resnet18(weights=None)
        with patch("models.spatial_encoder.resnet18", side_effect=factory), \
             patch("models.frequency_encoder_frozen.resnet18", side_effect=factory), \
             patch("models.frequency_encoder_pretrained.resnet18", side_effect=factory):
            b2 = BaselineB2()
            b = BaselineB(PretrainedFrequencyEncoder(feature_dim=256, freeze_backbone=False),
                          "normalized")
        return b2, b

    def test_same_fusion_and_forward_as_b_for_identical_weights(self):
        b2, b = self.models()
        self.assertIsInstance(b2.frequency_encoder, FrozenFrequencyEncoder)
        self.assertEqual(b2.classifier.in_features, 768)
        self.assertEqual(b2.classifier.out_features, 1)
        self.assertEqual(set(b2.state_dict()), set(b.state_dict()))
        b2.load_state_dict(b.state_dict())
        b2.eval()
        b.eval()
        images = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.equal(b2(images), b(images)))

    def test_only_classifier_trains_and_encoders_stay_in_eval(self):
        b2, _ = self.models()
        b2.train()
        self.assertFalse(b2.spatial_encoder.training)
        self.assertFalse(b2.frequency_encoder.training)
        self.assertFalse(b2.frequency_encoder.backbone.training)
        self.assertTrue(all(not p.requires_grad for p in b2.spatial_encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in b2.frequency_encoder.parameters()))
        self.assertTrue(all(p.requires_grad for p in b2.classifier.parameters()))
        logits = b2(torch.rand(2, 3, 64, 64))
        logits.square().mean().backward()
        self.assertEqual(tuple(logits.shape), (2, 1))
        self.assertTrue(all(p.grad is None for p in b2.spatial_encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in b2.frequency_encoder.parameters()))
        self.assertIsNotNone(b2.classifier.weight.grad)

    def test_training_epoch_accepts_frozen_frequency_branch(self):
        b2, _ = self.models()
        optimizer = torch.optim.Adam(b2.classifier.parameters(), lr=0.0009)
        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "label": torch.tensor([0, 1]),
            "path": ["archive/a.png", "archive/b.png"],
            "generator": ["real", "fake"],
        }
        metrics, rows = run_epoch_frozen(
            b2, [batch], torch.device("cpu"), optimizer, 1.0,
            prediction_context={"split": "train_sampled", "seed": 123},
        )
        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["model_name"], "baseline_b_2")
        self.assertTrue(all(p.grad is None for p in b2.frequency_encoder.parameters()))

    def test_protocol_changes_only_new_model_metadata_and_freezing(self):
        protocol = load_protocol("configs/baseline_b_2_lr9e-4.json")
        self.assertEqual(protocol["architecture"], ARCHITECTURE)
        self.assertTrue(protocol["freeze_frequency_encoder"])
        self.assertEqual(protocol["learning_rate"], 0.0009)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            protocol["batch_size"] = 16
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch_size"):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()
