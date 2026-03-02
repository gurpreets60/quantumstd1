"""
Random Filter CNN — fixed random convolutional filters + trainable classifier head.

QuanvNet analog: quantum convolution applies fixed random quantum circuits to
image patches, extracting quantum features. Here, random (frozen) CNN filters
extract classical random features; only the classifier head is trained.

Frozen random filters are surprisingly effective: random projections capture
diverse features, and the trainable head learns to combine them optimally.
"""

import torch.nn as nn
import torch.nn.functional as F
from .torch_base import TorchModel


class RandomFilterCNNModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("RANDOM_FILTER_CNN", epochs=epochs, img_shape=(1, 28, 28))

    def _build_net(self, num_classes):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                # Fixed random conv filters — NOT trained (frozen after random init)
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
                nn.init.normal_(self.conv1.weight)    # random normal init
                nn.init.normal_(self.conv2.weight)
                self.conv1.weight.requires_grad = False    # freeze: no gradient updates
                self.conv2.weight.requires_grad = False

                # Only the classification head is trainable
                self.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(4),           # -> (N, 64, 4, 4)
                    nn.Flatten(),
                    nn.Linear(64 * 16, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x):
                x = F.relu(self.conv1(x))    # random feature extraction
                x = F.max_pool2d(x, 2)       # 28x28 -> 14x14
                x = F.relu(self.conv2(x))    # second random feature layer
                return self.head(x)          # trainable head classifies

        return Net()
