import torch

from models.frequency_only import FrequencyOnlyModel


model = FrequencyOnlyModel(
    feature_dim=256,
    freeze_backbone=False
)

x = torch.randn(
    4,
    3,
    224,
    224
)

logits = model(x)

print("Input shape:", x.shape)
print("Output shape:", logits.shape)