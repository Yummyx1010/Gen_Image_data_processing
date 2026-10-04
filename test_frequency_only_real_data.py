import torch
from torch.utils.data import DataLoader

from dataset import GenImageDataset, transform, DATASET_ROOT
from models.frequency_only import FrequencyOnlyModel


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
# 2. Get one batch
# ==========================

batch = next(iter(loader))

images = batch["image"]

labels = (
    batch["label"]
    .float()
    .unsqueeze(1)
)


# ==========================
# 3. Build model
# ==========================

model = FrequencyOnlyModel(
    feature_dim=256,
    freeze_backbone=False
)

model.train()


# ==========================
# 4. Forward pass
# ==========================

logits = model(images)

print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("Logit shape:", logits.shape)


# ==========================
# 5. Loss
# ==========================

criterion = torch.nn.BCEWithLogitsLoss()

loss = criterion(
    logits,
    labels
)

print("Loss:", loss.item())


# ==========================
# 6. Backward
# ==========================

loss.backward()


# ==========================
# 7. Gradient check
# ==========================

print("\nGradient check:")

all_ok = True

for name, param in model.named_parameters():

    if param.requires_grad:

        has_grad = param.grad is not None

        print(
            name,
            "->",
            has_grad
        )

        if not has_grad:
            all_ok = False


print(
    "\nAll trainable parameters have gradients:",
    all_ok
)