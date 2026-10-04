import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class PretrainedFrequencyEncoder(nn.Module):
    def __init__(self, feature_dim=256, freeze_backbone=False):
        super().__init__()

        self.feature_dim = feature_dim

        # For reversing ImageNet normalization
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Load pretrained ResNet-18
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)

        # Remove original ImageNet classifier
        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # ResNet-18 feature dimension
        self.backbone_dim = 512

        # Optional freezing
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Project 512-D pretrained feature to 256-D
        self.fc = nn.Linear(
            self.backbone_dim,
            feature_dim
        )

    def denormalize(self, x):
        """
        Reverse ImageNet normalization approximately
        back to [0, 1].
        """
        x = x * self.imagenet_std + self.imagenet_mean
        return torch.clamp(x, 0.0, 1.0)

    def get_frequency_spectrum(self, x):
        """
        Input:
            x: [B, 3, 224, 224]

        Output:
            spectrum: [B, 1, 224, 224]
        """

        # Restore image intensity range
        x = self.denormalize(x)

        # RGB -> grayscale / luminance
        x = (
            0.299 * x[:, 0:1] +
            0.587 * x[:, 1:2] +
            0.114 * x[:, 2:3]
        )

        # 2D FFT
        fft = torch.fft.fft2(
            x,
            dim=(-2, -1)
        )

        # FFT shift
        fft = torch.fft.fftshift(
            fft,
            dim=(-2, -1)
        )

        # Magnitude
        magnitude = torch.abs(fft)

        # Log-magnitude spectrum
        spectrum = torch.log1p(magnitude)

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

        # Pretrained ResNet expects 3 channels
        spectrum_3ch = spectrum.repeat(
            1, 3, 1, 1
        )

        # [B, 512, 1, 1]
        features = self.backbone(
            spectrum_3ch
        )

        # [B, 512]
        features = torch.flatten(
            features,
            1
        )

        # [B, 256]
        features = self.fc(
            features
        )

        return features