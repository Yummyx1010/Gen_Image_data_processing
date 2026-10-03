from torch.utils.data import DataLoader

from dataset import GenImageDataset, transform, DATASET_ROOT
from models.frequency_encoder import FrequencyEncoder


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

batch = next(iter(loader))

images = batch["image"]

model = FrequencyEncoder(feature_dim=256)

spectrum = model.get_frequency_spectrum(images)
features = model(images)

print("Real input shape:", images.shape)
print("Spectrum shape:", spectrum.shape)
print("Feature shape:", features.shape)
print("Labels:", batch["label"])
print("Generators:", batch["generator"])