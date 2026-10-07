

import torch
from models.baseline_a import BaselineA


class FrozenBaselineA(BaselineA):
    def __init__(self):
        super().__init__()
        self.spatial_encoder.requires_grad_(False)
        self.spatial_encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.spatial_encoder.eval()
        return self

    def forward(self, images):
        with torch.no_grad():
            features = self.spatial_encoder(images)
        return self.classifier(features)  # M2's nn.Linear(512, 1), [B,1] logits
