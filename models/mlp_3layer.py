"""
3-hidden-layer MLP — deeper fully-connected network for MNIST.

Deeper VQC/QNN analog with more variational layers and expressibility.
784 -> 512 -> 256 -> 128 -> num_classes.
"""

import torch.nn as nn
from .torch_base import TorchModel


class MLP3LayerModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("MLP_3LAYER", epochs=epochs, img_shape=(784,))  # flat input

    def _build_net(self, num_classes):
        return nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),          # dropout before final layer
            nn.Linear(128, num_classes),
        )
