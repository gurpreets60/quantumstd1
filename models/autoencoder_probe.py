"""
Autoencoder + Linear Probe — two-stage: unsupervised pretraining then supervised.

QAE (Quantum Autoencoder) analog: quantum circuit compresses a quantum state
to a smaller register (latent), then reconstructs it. Here an MLP autoencoder
learns a compressed 64-dim latent space, then a linear probe classifies.

Stage 1: Autoencoder minimizes MSE(x, decode(encode(x))) — no labels used.
Stage 2: Linear probe trained on frozen encoder output — uses labels.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .torch_base import TorchModel


class _Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, latent_dim), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, z):
        return self.net(z)


class AutoencoderProbeModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("AUTOENCODER_PROBE", epochs=epochs, img_shape=(784,))
        self.latent_dim = 64
        self.encoder = self.decoder = self.probe = None

    def _build_net(self, num_classes):
        self.encoder = _Encoder(self.latent_dim)
        self.decoder = _Decoder(self.latent_dim)
        self.probe   = nn.Linear(self.latent_dim, num_classes)
        return self.encoder   # stored as self.net (unused directly)

    def train(self, X, y):
        num_classes = len(np.unique(y))
        self._build_net(num_classes)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.probe.to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dl  = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        # Stage 1: autoencoder reconstruction (unsupervised — no labels)
        ae_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr
        )
        print("  Stage 1: autoencoder pretraining")
        for ep in range(1, self.epochs + 1):
            total = 0.0
            for X_b, _ in dl:
                X_b = X_b.to(self.device)
                ae_opt.zero_grad()
                recon = self.decoder(self.encoder(X_b))
                loss  = F.mse_loss(recon, X_b)
                loss.backward(); ae_opt.step()
                total += loss.item() * len(X_b)
            print(f"    Epoch {ep}/{self.epochs}  recon_loss={total/len(X):.4f}")

        # Stage 2: linear probe on frozen encoder features (supervised)
        self.encoder.eval()
        probe_opt = torch.optim.Adam(self.probe.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        print("  Stage 2: linear probe on latent features")
        for ep in range(1, self.epochs + 1):
            total, correct = 0.0, 0
            for X_b, y_b in dl:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                probe_opt.zero_grad()
                with torch.no_grad():
                    z = self.encoder(X_b)        # frozen encoder: no gradient
                logits = self.probe(z)
                loss   = ce(logits, y_b)
                loss.backward(); probe_opt.step()
                total   += loss.item() * len(X_b)
                correct += (logits.argmax(1) == y_b).sum().item()
            print(f"    Epoch {ep}/{self.epochs}  loss={total/len(X):.4f}  acc={correct/len(X):.4f}")

    def predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.encoder.eval(); self.probe.eval()
        with torch.no_grad():
            logits = self.probe(self.encoder(X_t))
        return F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
