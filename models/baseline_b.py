"""Baseline B with frozen spatial features and trainable frequency features."""


import torch
from torch import nn
from models.spatial_encoder import SpatialEncoder


class BaselineB(nn.Module):
    """512 + 256 -> Linear(768, 1), matching M2's single-linear-layer head.

    frequency_encoder must include FFT/log-spectrum preprocessing, produce
    [B,256], and remain connected to autograd. A shared spatial encoder may
    be injected for controlled tests.
    """

    def __init__(self, frequency_encoder, frequency_input, spatial_encoder=None):
        super().__init__()
        if frequency_input not in {"normalized", "rgb01"}:
            raise ValueError("frequency_input must be normalized or rgb01")
        if not isinstance(frequency_encoder, nn.Module):
            raise TypeError("frequency_encoder must be an nn.Module returning [B,256]")
        if not any(p.requires_grad for p in frequency_encoder.parameters()):
            raise ValueError("Baseline B requires a trainable frequency encoder")
        self.spatial_encoder = spatial_encoder if spatial_encoder is not None else SpatialEncoder(freeze=True)
        self.spatial_encoder.requires_grad_(False)
        self.spatial_encoder.eval()
        self.frequency_encoder = frequency_encoder
        self.frequency_input = frequency_input
        self.classifier = nn.Linear(512 + 256, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.spatial_encoder.eval()
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
        frequency_input = images * self.std + self.mean if self.frequency_input == "rgb01" else images
        frequency = self.frequency_encoder(frequency_input)
        self._check_features(spatial, images.shape[0], 512, "Spatial encoder")
        self._check_features(frequency, images.shape[0], 256, "Frequency encoder")
        if spatial.device != frequency.device or spatial.dtype != frequency.dtype:
            raise ValueError("Spatial/frequency features must use the same device and dtype")
        if self.training and torch.is_grad_enabled() and not frequency.requires_grad:
            raise RuntimeError("Detached frequency features cannot train the frequency branch end-to-end")
        return self.classifier(torch.cat([spatial, frequency], dim=1))
