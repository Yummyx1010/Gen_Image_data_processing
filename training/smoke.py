"""Offline fixtures only: NOT a trained spatial model or M3 frequency model."""

from contextlib import contextmanager
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import resnet18


class FrequencyStub(nn.Module):
    """Trainable RGB fixture returning 256 values; no FFT is performed."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 256))

    def forward(self, images):
        return self.encoder(images)


class SyntheticDataset(Dataset):
    def __init__(self, split, seed):
        size = 8 if split == "train" else 4
        offset = {"train": 10, "validation": 20, "test": 30}[split]
        generator = torch.Generator().manual_seed(seed + offset)
        images = torch.rand(size, 3, 64, 64, generator=generator)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.images = (images - mean) / std
        self.split = split

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {"image": self.images[index], "label": index % 2,
                "generator": "smoke_fake" if index % 2 else "smoke_real",
                "path": f"synthetic/{self.split}/{index}"}


@contextmanager
def offline_spatial_initialization():
    """Exercise M2's real ResNet-18 architecture without downloading weights.

    The patch is scoped to construction, restores the original function, and is
    entered only when smoke=True. Formal runs always load M2's pretrained weights.
    """
    with patch("models.spatial_encoder.resnet18", side_effect=lambda **kwargs: resnet18(weights=None)):
        yield
