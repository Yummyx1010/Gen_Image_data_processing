import torch

from models.frequency_encoder_pretrained import (
    PretrainedFrequencyEncoder
)


model = PretrainedFrequencyEncoder(
    feature_dim=256,
    freeze_backbone=False
)

x = torch.randn(
    4,
    3,
    224,
    224
)

spectrum = model.get_frequency_spectrum(x)

features = model(x)

print("Input shape:", x.shape)
print("Spectrum shape:", spectrum.shape)
print("Feature shape:", features.shape)

print(
    "Spectrum NaN:",
    torch.isnan(spectrum).any()
)

print(
    "Spectrum Inf:",
    torch.isinf(spectrum).any()
)