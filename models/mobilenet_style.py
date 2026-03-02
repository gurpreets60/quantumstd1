"""
MobileNet-style CNN — depthwise separable convolutions for parameter efficiency.

PV-QNN (Parameter-efficient Variational QNN) analog: depthwise separable convs
factorize a full convolution into two cheaper operations:
  - Depthwise: each input channel convolved with its own 3x3 filter (no cross-channel mixing)
  - Pointwise: 1x1 conv mixes channels (cross-channel information exchange)
This factorization reduces parameters ~8-9x vs standard conv with similar accuracy.
"""

import torch.nn as nn
from .torch_base import TorchModel


def ds_conv(in_ch, out_ch, stride=1):
    """Depthwise separable conv block: depthwise + pointwise + BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                  groups=in_ch, bias=False),   # depthwise: each channel independently
        nn.Conv2d(in_ch, out_ch, 1, bias=False),              # pointwise: mix channels
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MobileNetStyleModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("MOBILENET_STYLE", epochs=epochs, img_shape=(1, 28, 28))

    def _build_net(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),       # standard conv to expand channels
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            ds_conv(16, 32),                                   # 28x28
            ds_conv(32, 64, stride=2),                         # 28x28 -> 14x14
            ds_conv(64, 128, stride=2),                        # 14x14 -> 7x7
            nn.AdaptiveAvgPool2d(1),                           # 7x7 -> 1x1
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )
