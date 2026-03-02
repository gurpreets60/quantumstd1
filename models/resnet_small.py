"""
Small ResNet — residual skip connections for improved gradient flow.

ResBlocks: two Conv-BN-ReLU layers + identity skip connection.
Skip connections allow gradients to bypass layers (avoids vanishing gradients).
Hybrid QCNN analog: skip connections in quantum circuits help preserve quantum info.
"""

import torch
import torch.nn as nn
from .torch_base import TorchModel


class ResBlock(nn.Module):
    """Two Conv-BN-ReLU layers with an identity skip connection."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))   # residual: add input to block output


class ResNetSmallModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("RESNET_SMALL", epochs=epochs, img_shape=(1, 28, 28))

    def _build_net(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock(32),
            nn.MaxPool2d(2),                    # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),
            nn.MaxPool2d(2),                    # 14x14 -> 7x7
            nn.AdaptiveAvgPool2d(1),            # 7x7 -> 1x1 global average pool
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )
