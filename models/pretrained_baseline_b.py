"""Spatial-frequency fusion with the pretrained frequency encoder from main."""
import torch
from torch import nn

from models.frequency_encoder_pretrained import PretrainedFrequencyEncoder
from models.spatial_encoder import SpatialEncoder


class NonlinearClassifier(nn.Sequential):
    """Identical classifier topology for all three feature combinations."""

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


class PretrainedBaselineB(nn.Module):
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