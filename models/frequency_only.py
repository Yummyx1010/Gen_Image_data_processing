import torch
import torch.nn as nn

from models.frequency_encoder_pretrained import PretrainedFrequencyEncoder


class FrequencyOnlyModel(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        freeze_backbone=False
    ):
        super().__init__()

        # Pretrained frequency encoder
        self.frequency_encoder = PretrainedFrequencyEncoder(
            feature_dim=feature_dim,
            freeze_backbone=freeze_backbone
        )

        # Binary classifier
        self.classifier = nn.Linear(
            feature_dim,
            1
        )

    def forward(self, x):
        # [B, 3, 224, 224]
        frequency_features = self.frequency_encoder(x)

        # [B, 256] -> [B, 1]
        logits = self.classifier(
            frequency_features
        )

        return logits