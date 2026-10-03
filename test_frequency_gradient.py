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

model.train()

features = model(images)

# Simple dummy loss only for checking gradient flow
loss = features.mean()

loss.backward()

print("Feature shape:", features.shape)
print("Loss:", loss.item())

print("\nGradient check:")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, "->", param.grad is not None)