"""The historical pretrained B and the current presentation-aligned B."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from models.baseline_a_adapter import FrozenBaselineA
from models.baseline_b import BaselineB
from models.frequency_encoder import FrequencyEncoder
from models.frequency_encoder_pretrained import PretrainedFrequencyEncoder
from models.frequency_only import FrequencyOnlyModel
from models.pretrained_baseline_b import PretrainedBaselineB
from train_final import CURRENT_B_ARCHITECTURE, load_regularized_checkpoint
from training.train import seed_everything


class ControlledBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_models(self):
        # Keep the real ResNet-18 structure while avoiding downloads in tests.
        factory = lambda **kwargs: resnet18(weights=None)
        with patch("models.spatial_encoder.resnet18", side_effect=factory), \
             patch("models.frequency_encoder_pretrained.resnet18", side_effect=factory):
            seed_everything(42)
            a = FrozenBaselineA()
            seed_everything(42)
            b = PretrainedBaselineB(seed=42)
            seed_everything(42)
            frequency_only = FrequencyOnlyModel()
        return a, b, frequency_only

    def test_shared_spatial_path_and_frequency_initialization(self):
        a, b, frequency_only = self.make_models()
        self.assertIsInstance(b.classifier, nn.Linear)
        self.assertEqual(b.classifier.in_features, 512)
        self.assertEqual(b.frequency_classifier.in_features, 256)
        self.assertIsNone(b.frequency_classifier.bias)
        for key, value in a.spatial_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, b.spatial_encoder.state_dict()[key]), key)
        for key, value in a.classifier.state_dict().items():
            self.assertTrue(torch.equal(value, b.classifier.state_dict()[key]), key)
        for key, value in frequency_only.frequency_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, b.frequency_encoder.state_dict()[key]), key)

    def test_frequency_path_can_be_removed_without_changing_a_prediction(self):
        a, b, _ = self.make_models()
        a.eval()
        b.eval()
        with torch.no_grad():
            b.frequency_classifier.weight.zero_()
            images = torch.rand(2, 3, 64, 64)
            self.assertTrue(torch.equal(a(images), b(images)))

    def test_only_frequency_and_linear_heads_receive_gradients(self):
        _, b, _ = self.make_models()
        b.train()
        logits = b(torch.rand(2, 3, 64, 64))
        logits.square().mean().backward()
        self.assertEqual(tuple(logits.shape), (2, 1))
        self.assertFalse(b.spatial_encoder.training)
        self.assertTrue(all(p.grad is None for p in b.spatial_encoder.parameters()))
        self.assertIsNotNone(b.classifier.weight.grad)
        self.assertIsNotNone(b.frequency_classifier.weight.grad)
        self.assertTrue(any(p.grad is not None for p in b.frequency_encoder.parameters()))

    def test_current_b_uses_m3_encoder_and_real_768_feature_concat(self):
        requested_weights = []
        def factory(**kwargs):
            requested_weights.append(kwargs["weights"])
            return resnet18(weights=None)
        with patch("models.spatial_encoder.resnet18", side_effect=factory), \
             patch("models.frequency_encoder_pretrained.resnet18", side_effect=factory):
            b = BaselineB(PretrainedFrequencyEncoder(feature_dim=256, freeze_backbone=False),
                          "normalized")
        self.assertEqual(requested_weights, [ResNet18_Weights.DEFAULT] * 2)
        self.assertIsInstance(b.frequency_encoder, PretrainedFrequencyEncoder)
        self.assertEqual(b.classifier.in_features, 768)
        classifier_inputs = []
        hook = b.classifier.register_forward_pre_hook(
            lambda module, args: classifier_inputs.append(tuple(args[0].shape))
        )
        try:
            logits = b(torch.rand(2, 3, 64, 64))
        finally:
            hook.remove()
        self.assertEqual(classifier_inputs, [(2, 768)])
        self.assertEqual(tuple(logits.shape), (2, 1))
        logits.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in b.spatial_encoder.parameters()))
        self.assertIsNotNone(b.classifier.weight.grad)
        self.assertTrue(any(p.grad is not None for p in b.frequency_encoder.backbone.parameters()))
        self.assertIsNotNone(b.frequency_encoder.fc.weight.grad)

    def test_current_b_checkpoint_loads_by_its_own_architecture_id(self):
        factory = lambda **kwargs: resnet18(weights=None)
        with patch("models.spatial_encoder.resnet18", side_effect=factory), \
             patch("models.frequency_encoder_pretrained.resnet18", side_effect=factory):
            b = BaselineB(PretrainedFrequencyEncoder(feature_dim=256, freeze_backbone=False),
                          "normalized")
            checkpoint = {
                "format_version": 2,
                "architecture": CURRENT_B_ARCHITECTURE,
                "model_name": "baseline_b",
                "protocol": {"architecture": CURRENT_B_ARCHITECTURE,
                             "freeze_frequency_backbone": False},
                "model_state_dict": b.state_dict(),
            }
            with patch("train_final.torch.load", return_value=checkpoint):
                loaded, _ = load_regularized_checkpoint("unused.pt")
        self.assertIsInstance(loaded, BaselineB)
        self.assertIsInstance(loaded.frequency_encoder, PretrainedFrequencyEncoder)
        self.assertEqual(loaded.classifier.in_features, 768)

    def test_legacy_cnn_b_checkpoint_still_loads(self):
        factory = lambda **kwargs: resnet18(weights=None)
        with patch("models.spatial_encoder.resnet18", side_effect=factory):
            b = BaselineB(FrequencyEncoder(feature_dim=256), "normalized")
            checkpoint = {
                "format_version": 2,
                "architecture": "frequency_cnn_concat_v1",
                "model_name": "baseline_b",
                "protocol": {"architecture": "frequency_cnn_concat_v1"},
                "model_state_dict": b.state_dict(),
            }
            with patch("train_final.torch.load", return_value=checkpoint):
                loaded, _ = load_regularized_checkpoint("unused.pt")
        self.assertIsInstance(loaded.frequency_encoder, FrequencyEncoder)


if __name__ == "__main__":
    unittest.main()
