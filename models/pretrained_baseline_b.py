"""Controlled spatial-frequency model and the legacy nonlinear fusion model."""
import torch
from torch import nn

from models.frequency_encoder_pretrained import PretrainedFrequencyEncoder
from models.spatial_encoder import SpatialEncoder


class NonlinearClassifier(nn.Sequential):
    """Classifier retained for historical nonlinear fusion checkpoints."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),
        )


class LegacyPretrainedBaselineB(nn.Module):
    """Original nonlinear fusion architecture for loading historical checkpoints."""

    def __init__(self, hidden_dim=128, dropout=0.3, freeze_backbone=False, seed=None):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(freeze=True)
        self.spatial_encoder.requires_grad_(False)
        self.spatial_encoder.eval()
        # Match the frequency-only branch's initialization for the same seed.
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.default_generator.manual_seed(seed)
            self.frequency_encoder = PretrainedFrequencyEncoder(
                feature_dim=256, freeze_backbone=freeze_backbone
            )
        self.classifier = NonlinearClassifier(768, hidden_dim, dropout)

    def train(self, mode=True):
        super().train(mode)
        self.spatial_encoder.eval()
        return self

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected normalized RGB images [B,3,H,W]")
        with torch.no_grad():
            spatial = self.spatial_encoder(images)
        frequency = self.frequency_encoder(images)
        if spatial.shape != (images.shape[0], 512):
            raise ValueError("Spatial encoder must return [B,512]")
        if frequency.shape != (images.shape[0], 256):
            raise ValueError("Frequency encoder must return [B,256]")
        fused = torch.cat((spatial, frequency), dim=1)
        if not torch.isfinite(fused).all():
            raise ValueError("Non-finite fused features")
        return self.classifier(fused)


class PretrainedBaselineB(nn.Module):
    """Baseline A's linear spatial path plus a trainable frequency contribution.

    Constructing the spatial classifier before the frequency branch makes its
    initialization identical to Baseline A when both use the same global seed.
    The frequency classifier has no bias, so the spatial path retains A's
    complete 512-to-1 classifier, including its bias.
    """

    def __init__(self, freeze_backbone=False, seed=None):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(freeze=True)
        self.spatial_encoder.requires_grad_(False)
        self.spatial_encoder.eval()
        self.classifier = nn.Linear(self.spatial_encoder.feature_dim, 1)

        # Match the frequency-only encoder's initialization at the same seed.
        with torch.random.fork_rng(devices=[]):
            if seed is not None:
                torch.default_generator.manual_seed(seed)
            self.frequency_encoder = PretrainedFrequencyEncoder(
                feature_dim=256, freeze_backbone=freeze_backbone
            )
        self.frequency_classifier = nn.Linear(256, 1, bias=False)

    def train(self, mode=True):
        super().train(mode)
        self.spatial_encoder.eval()
        return self

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected normalized RGB images [B,3,H,W]")
        with torch.no_grad():
            spatial = self.spatial_encoder(images)
        frequency = self.frequency_encoder(images)
        if spatial.shape != (images.shape[0], 512):
            raise ValueError("Spatial encoder must return [B,512]")
        if frequency.shape != (images.shape[0], 256):
            raise ValueError("Frequency encoder must return [B,256]")
        if not torch.isfinite(spatial).all() or not torch.isfinite(frequency).all():
            raise ValueError("Non-finite spatial or frequency features")
        return self.classifier(spatial) + self.frequency_classifier(frequency)
