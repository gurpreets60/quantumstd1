"""
1-hidden-layer MLP — simplest neural network baseline.

VQC / QNN classical analog: a single variational layer transforms features
then classifies. 784 -> 512 -> num_classes.
"""

import torch.nn as nn
from .torch_base import TorchModel


class MLP1LayerModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("MLP_1LAYER", epochs=epochs, img_shape=(784,))  # flat input

    def _build_net(self, num_classes):
        return nn.Sequential(
            nn.Linear(784, 512),      # single hidden layer
            nn.ReLU(),
            nn.Dropout(0.3),          # regularization: randomly zero 30% of activations
            nn.Linear(512, num_classes),
        )
