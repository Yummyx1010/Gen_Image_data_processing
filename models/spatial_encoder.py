import torch
import torch.nn as nn
from pyexpat import features
from torchvision.models import resnet18, ResNet18_Weights

class SpatialEncoder(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

        ## Load pretrained ResNet-18
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights = weights)

        ## Remove the original ImageNet classification head
        self.encoder = nn.Sequential(*list(model.children())[:-1])

        ## Resnet-18 output feature dimension
        self.feature_dim = 512

        ## Freeze encoder parameters for the primary controlled experiment
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)

        ## [B, 512, 1, 1] -> [B, 512]
        features = torch.flatten(features, 1)

        return features