"""
LeNet-5 — original 1998 CNN (LeCun et al.), designed specifically for MNIST.

Architecture: Conv(6) -> Pool -> Conv(16) -> Pool -> FC(120) -> FC(84) -> out.
Uses tanh activations and average pooling (as in the original paper).
QCQ-CNN analog: convolutional feature extraction + dense classification layers.
~99% on MNIST with proper training.
"""

import torch.nn as nn
from .torch_base import TorchModel


class LeNet5Model(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("LENET5", epochs=epochs, img_shape=(1, 28, 28))

    def _build_net(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),   # C1: 6 feature maps, 28x28
            nn.Tanh(),
            nn.AvgPool2d(2),                              # S2: subsample to 14x14
            nn.Conv2d(6, 16, kernel_size=5),              # C3: 16 feature maps, 10x10
            nn.Tanh(),
            nn.AvgPool2d(2),                              # S4: subsample to 5x5
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),                  # C5: fully connected
            nn.Tanh(),
            nn.Linear(120, 84),                           # F6
            nn.Tanh(),
            nn.Linear(84, num_classes),                   # output layer
        )
