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

batch = next(iter(loader))

images = batch["image"]


# ==========================
# 2. Build pretrained model
# ==========================

model = PretrainedFrequencyEncoder(
    feature_dim=256,
    freeze_backbone=False
)

model.train()


# ==========================
# 3. Forward pass
# ==========================

features = model(images)

print("Feature shape:", features.shape)


# ==========================
# 4. Dummy loss
# ==========================

loss = features.mean()

print("Loss:", loss.item())


# ==========================
# 5. Backward
# ==========================

loss.backward()


# ==========================
# 6. Gradient check
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


# ==========================
# 7. Final result
# ==========================

print("\nAll trainable parameters have gradients:", all_ok)