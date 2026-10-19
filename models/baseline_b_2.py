"""Baseline B_2: the existing 512+256 fusion with both encoders frozen."""

import torch
from torch import nn

from models.frequency_encoder_frozen import FrozenFrequencyEncoder
from models.spatial_encoder import SpatialEncoder


class BaselineB2(nn.Module):
    """Keep Baseline B's inputs and Linear(768, 1) head; train only that head.

    M3's FrozenFrequencyEncoder freezes its ResNet backbone but leaves its
    512-to-256 projection trainable. B_2 freezes that projection as well.
    With no supplied frequency checkpoint, the projection is seeded but
    randomly initialized; this is a frozen-feature ablation, not a pretrained
    frequency-only model.
    """

    def __init__(self, spatial_encoder=None, frequency_encoder=None):
        super().__init__()
        self.spatial_encoder = spatial_encoder if spatial_encoder is not None else SpatialEncoder(freeze=True)
        self.frequency_encoder = (
            frequency_encoder if frequency_encoder is not None
            else FrozenFrequencyEncoder(feature_dim=256)
        )
        self.spatial_encoder.requires_grad_(False)
        self.frequency_encoder.requires_grad_(False)
        self.spatial_encoder.eval()
        self.frequency_encoder.eval()
        self.classifier = nn.Linear(512 + 256, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.spatial_encoder.eval()
        self.frequency_encoder.eval()
        return self

    @staticmethod
    def _check_features(value, batch_size, dimension, branch):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch_size, dimension):
            raise ValueError(f"{branch} must return a Tensor [{batch_size},{dimension}]")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{branch} features must be finite floating-point values")

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected M1 normalized RGB images [B,3,H,W]")
        with torch.no_grad():
            spatial = self.spatial_encoder(images)
            frequency = self.frequency_encoder(images)
        self._check_features(spatial, images.shape[0], 512, "Spatial encoder")
        self._check_features(frequency, images.shape[0], 256, "Frequency encoder")
        if spatial.device != frequency.device or spatial.dtype != frequency.dtype:
            raise ValueError("Spatial/frequency features must use the same device and dtype")
        return self.classifier(torch.cat([spatial, frequency], dim=1))
