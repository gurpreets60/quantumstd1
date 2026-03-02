"""
Small DenseNet — dense connections: each layer receives all preceding feature maps.

Growth rate k=16: each DenseLayer adds 16 new feature maps, concatenated to input.
After 4 layers: 32 + 4*16 = 96 channels total.
Dense connectivity encourages feature reuse and improves gradient flow.
"""

import torch
import torch.nn as nn
from .torch_base import TorchModel


class DenseLayer(nn.Module):
    """BN -> ReLU -> Conv. Output concatenated with input (dense connection)."""

    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)   # dense connection: concat output to input


class DenseNetSmallModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("DENSENET_SMALL", epochs=epochs, img_shape=(1, 28, 28))

    def _build_net(self, num_classes):
        growth = 16          # channels added per dense layer
        layers = [nn.Conv2d(1, 32, 3, padding=1, bias=False)]
        ch = 32
        for _ in range(4):   # 4 dense layers: ch grows 32->48->64->80->96
            layers.append(DenseLayer(ch, growth))
            ch += growth
        layers += [
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),             # pool spatial dims to 4x4
            nn.Flatten(),
            nn.Linear(ch * 16, num_classes),     # ch * 4*4 = 96*16 = 1536
        ]
        return nn.Sequential(*layers)
