"""
Shared PyTorch base for all deep learning classifiers in this pipeline.

Subclasses implement _build_net(num_classes) -> nn.Module.
The base handles: device detection, DataLoader creation, standard
cross-entropy training loop, and softmax predict_proba.

Models with custom training loops (VAE, SimCLR, CapsNet, Autoencoder)
subclass TorchModel but override train() while still inheriting
_to_loader(), device, and predict_proba().
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseModel


class TorchModel(BaseModel):
    """
    PyTorch base: subclass and implement _build_net(num_classes) -> nn.Module.
    img_shape controls how flat (N, pixels) input is reshaped for the network:
      (1, 28, 28) for CNN-based models
      (28, 28)    for LSTM (28 timesteps of 28 features)
      (784,)      for flat MLP models
    """

    def __init__(self, name, epochs=5, batch_size=256, lr=1e-3, img_shape=(1, 28, 28)):
        super().__init__(name)
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.img_shape  = img_shape
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net        = None  # built lazily in train() once num_classes is known

    def _build_net(self, num_classes: int) -> nn.Module:
        """Return an nn.Module for num_classes outputs. Subclasses must override."""
        raise NotImplementedError

    def _to_loader(self, X, y=None, shuffle=False) -> DataLoader:
        """Reshape flat numpy (N, pixels) -> DataLoader with self.img_shape."""
        X_t = torch.tensor(X, dtype=torch.float32).reshape(-1, *self.img_shape)
        ds  = TensorDataset(X_t, torch.tensor(y, dtype=torch.long)) if y is not None \
              else TensorDataset(X_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, X_train, y_train):
        """Detect num_classes, build net, then train with Adam + cross-entropy."""
        num_classes = len(np.unique(y_train))
        self.net    = self._build_net(num_classes).to(self.device)
        loader      = self._to_loader(X_train, y_train, shuffle=True)
        opt         = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn     = nn.CrossEntropyLoss()

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            total_loss, correct = 0.0, 0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                opt.zero_grad()
                logits = self.net(X_b)
                loss   = loss_fn(logits, y_b)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(X_b)
                correct    += (logits.argmax(1) == y_b).sum().item()
            print(f"  Epoch {epoch}/{self.epochs}  "
                  f"loss={total_loss/len(X_train):.4f}  "
                  f"acc={correct/len(X_train):.4f}")

    def predict_proba(self, X) -> np.ndarray:
        """Forward pass -> softmax probabilities (N, K)."""
        loader = self._to_loader(X, shuffle=False)
        self.net.eval()
        out = []
        with torch.no_grad():
            for (X_b,) in loader:
                logits = self.net(X_b.to(self.device))
                out.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
