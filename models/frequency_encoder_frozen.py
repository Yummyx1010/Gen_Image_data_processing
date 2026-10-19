import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class FrozenFrequencyEncoder(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()

        self.feature_dim = feature_dim

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Pretrained ResNet-18
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)

        # Remove original classifier
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.backbone_dim = 512

        # Freeze pretrained backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Projection layer remains trainable
        self.fc = nn.Linear(
            self.backbone_dim,
            feature_dim
        )

    def denormalize(self, x):
        x = x * self.imagenet_std + self.imagenet_mean
        return torch.clamp(x, 0.0, 1.0)

    def get_frequency_spectrum(self, x):
        # Restore RGB intensity
        x = self.denormalize(x)

        # RGB -> grayscale
        x = (
            0.299 * x[:, 0:1] +
            0.587 * x[:, 1:2] +
            0.114 * x[:, 2:3]
        )

        # FFT
        fft = torch.fft.fft2(
            x,
            dim=(-2, -1)
        )

        fft = torch.fft.fftshift(
            fft,
            dim=(-2, -1)
        )

        magnitude = torch.abs(fft)

        spectrum = torch.log1p(
            magnitude
        )

        # Per-image normalization
        mean = spectrum.mean(
            dim=(-2, -1),
            keepdim=True
        )

        std = spectrum.std(
            dim=(-2, -1),
            keepdim=True
        )

        spectrum = (
            spectrum - mean
        ) / (std + 1e-6)

        return spectrum

    def forward(self, x):
        # [B, 1, 224, 224]
        spectrum = self.get_frequency_spectrum(x)

        # Repeat to 3 channels
        spectrum = spectrum.repeat(
            1, 3, 1, 1
        )

        # Frozen backbone
        features = self.backbone(
            spectrum
        )

        # [B, 512, 1, 1] -> [B, 512]
        features = torch.flatten(
            features,
            1
        )

        # Trainable projection
        features = self.fc(
            features
        )

        return features