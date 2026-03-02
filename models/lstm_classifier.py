"""
LSTM Image Classifier — treats 28x28 image as a sequence of 28 row-vectors.

Each row of 28 pixels is one timestep; LSTM reads rows top-to-bottom.
LSTM hidden state aggregates spatial patterns sequentially across rows.
QTL (Quantum Transfer Learning) analog: sequential processing of feature vectors
through parameterized quantum layers, one per timestep.
"""

import torch.nn as nn
from .torch_base import TorchModel


class LSTMNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=28,    # 28 pixel values per row (features per timestep)
            hidden_size=128,  # LSTM hidden state dimensionality
            num_layers=2,     # stacked LSTM (2nd layer takes 1st layer's hidden state)
            batch_first=True, # input shape: (batch, seq_len, features)
            dropout=0.3,      # dropout between LSTM layers
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (N, 28, 28) — N images, each as 28 timesteps of 28 features
        out, _ = self.lstm(x)         # out: (N, 28, 128) — all hidden states
        return self.fc(out[:, -1, :]) # use last timestep's hidden state for classification


class LSTMClassifierModel(TorchModel):
    def __init__(self, epochs=10):
        # img_shape=(28, 28): reshape flat (N,784) to (N,28,28) — 28 timesteps of 28 pixels
        super().__init__("LSTM_CLASSIFIER", epochs=epochs, img_shape=(28, 28))

    def _build_net(self, num_classes):
        return LSTMNet(num_classes)
