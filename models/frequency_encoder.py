# models/frequency_encoder.py

import torch
import torch.nn as nn


class FrequencyEncoder(nn.Module):
    def __init__(self, feature_dim=256):
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

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(128, feature_dim)

    def denormalize(self, x):
        x = x * self.imagenet_std + self.imagenet_mean
        return torch.clamp(x, 0.0, 1.0)

    def get_frequency_spectrum(self, x):
        # x: [B, 3, 224, 224]

        # Restore image intensity range
        x = self.denormalize(x)

        # RGB -> grayscale
        x = (
            0.299 * x[:, 0:1] +
            0.587 * x[:, 1:2] +
            0.114 * x[:, 2:3]
        )

        # 2D FFT
        fft = torch.fft.fft2(x, dim=(-2, -1))

        # Move low-frequency component to centre
        fft = torch.fft.fftshift(fft, dim=(-2, -1))

        # Magnitude spectrum
        magnitude = torch.abs(fft)

        # Log magnitude
        spectrum = torch.log1p(magnitude)

        # Per-image normalization
        mean = spectrum.mean(dim=(-2, -1), keepdim=True)
        std = spectrum.std(dim=(-2, -1), keepdim=True)

        spectrum = (spectrum - mean) / (std + 1e-6)

        return spectrum

    def forward(self, x):
        spectrum = self.get_frequency_spectrum(x)

        features = self.encoder(spectrum)

        features = torch.flatten(features, 1)

        features = self.fc(features)

        return features