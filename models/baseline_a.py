import torch
import torch.nn as nn

from models.spatial_encoder import SpatialEncoder

class BaselineA(nn.Module):
    def __init__(self):
        super().__init__()
        ## Shared frozen spatial encoder
        self.spatial_encoder = SpatialEncoder(freeze=True)
        ## Trainable binary classifier
        self.classifier = nn.Linear(
            self.spatial_encoder.feature_dim,
            1
    )

    def forward(self, x):
        spatial_features = self.spatial_encoder(x)

        logits = self.classifier(spatial_features)

        return logits