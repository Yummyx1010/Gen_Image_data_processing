import torch
from torch.utils.data import DataLoader

from dataset import GenImageDataset, transform, DATASET_ROOT
from models.frequency_encoder_pretrained import PretrainedFrequencyEncoder


# ==========================
# 1. Load real dataset
# ==========================

dataset = GenImageDataset(
    csv_file="train.csv",
    root_dir=DATASET_ROOT,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# ==========================
# 2. Get one real batch
# ==========================

batch = next(iter(loader))

images = batch["image"]
labels = batch["label"]
generators = batch["generator"]


# ==========================
# 3. Build pretrained
#    frequency encoder
# ==========================

model = PretrainedFrequencyEncoder(
    feature_dim=256,
    freeze_backbone=False
)

model.eval()


# ==========================
# 4. Forward pass
# ==========================

with torch.no_grad():

    spectrum = model.get_frequency_spectrum(
        images
    )

    features = model(
        images
    )


# ==========================
# 5. Basic checks
# ==========================

print("Real input shape:")
print(images.shape)

print("\nSpectrum shape:")
print(spectrum.shape)

print("\nFeature shape:")
print(features.shape)

print("\nLabels:")
print(labels)

print("\nGenerators:")
print(generators)


# ==========================
# 6. Numerical checks
# ==========================

print(
    "\nSpectrum NaN:",
    torch.isnan(spectrum).any()
)

print(
    "Spectrum Inf:",
    torch.isinf(spectrum).any()
)

print(
    "Feature NaN:",
    torch.isnan(features).any()
)

print(
    "Feature Inf:",
    torch.isinf(features).any()
)


# ==========================
# 7. Feature statistics
# ==========================

print(
    "\nFeature mean:",
    features.mean().item()
)

print(
    "Feature std:",
    features.std().item()
)


print(
    "\nPretrained frequency encoder "
    "real-data test completed."
)