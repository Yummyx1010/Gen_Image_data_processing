import torch

from models.frequency_encoder import FrequencyEncoder


model = FrequencyEncoder(feature_dim=256)

# Fake batch with the same shape as the real DataLoader
x = torch.randn(4, 3, 224, 224)

# Test frequency spectrum
spectrum = model.get_frequency_spectrum(x)

# Test frequency encoder
features = model(x)

print("Input shape:", x.shape)
print("Spectrum shape:", spectrum.shape)
print("Feature shape:", features.shape)

print("Spectrum NaN:", torch.isnan(spectrum).any())
print("Spectrum Inf:", torch.isinf(spectrum).any())