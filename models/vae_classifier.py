"""
VAE + Linear Probe — Variational Autoencoder then linear classification.

VAE learns a probabilistic latent space by maximizing the ELBO:
  ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
     = reconstruction term - KL regularization to N(0,I) prior

QAE analog: quantum amplitude encoding + variational circuit maps quantum states
to a compressed latent register, similar to VAE's encoder-decoder structure.

Stage 1: train VAE on unlabeled data via ELBO.
Stage 2: freeze encoder, train linear probe on mean embedding mu(x).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .torch_base import TorchModel


class _VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.enc    = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
        self.mu_fc  = nn.Linear(256, latent_dim)   # mean of q(z|x)
        self.lv_fc  = nn.Linear(256, latent_dim)   # log-variance of q(z|x)
        self.dec    = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(),
                                    nn.Linear(256, 784), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu_fc(h), self.lv_fc(h)   # mu, log_var

    def forward(self, x):
        mu, lv = self.encode(x)
        std     = torch.exp(0.5 * lv)
        z       = mu + std * torch.randn_like(std)   # reparameterization trick
        return self.dec(z), mu, lv


class VAEClassifierModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("VAE_CLASSIFIER", epochs=epochs, img_shape=(784,))
        self.latent_dim = 32
        self.vae = self.probe = None

    def _build_net(self, num_classes):
        self.vae   = _VAE(self.latent_dim)
        self.probe = nn.Linear(self.latent_dim, num_classes)
        return self.vae

    def train(self, X, y):
        num_classes = len(np.unique(y))
        self._build_net(num_classes)
        self.vae.to(self.device); self.probe.to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dl  = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        # Stage 1: VAE pretraining via ELBO
        vae_opt = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        print("  Stage 1: VAE pretraining (ELBO = recon - KL)")
        for ep in range(1, self.epochs + 1):
            total = 0.0
            for X_b, _ in dl:
                X_b = X_b.to(self.device)
                vae_opt.zero_grad()
                recon, mu, lv = self.vae(X_b)
                recon_loss = F.mse_loss(recon, X_b, reduction="sum")
                kl_loss    = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
                loss = (recon_loss + kl_loss) / len(X_b)
                loss.backward(); vae_opt.step()
                total += loss.item() * len(X_b)
            print(f"    Epoch {ep}/{self.epochs}  elbo_loss={total/len(X):.2f}")

        # Stage 2: linear probe on frozen encoder's mean output
        self.vae.eval()
        probe_opt = torch.optim.Adam(self.probe.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        print("  Stage 2: linear probe on VAE latent mean")
        for ep in range(1, self.epochs + 1):
            total, correct = 0.0, 0
            for X_b, y_b in dl:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                probe_opt.zero_grad()
                with torch.no_grad():
                    mu, _ = self.vae.encode(X_b)   # use mean (deterministic at test time)
                logits  = self.probe(mu)
                loss    = ce(logits, y_b)
                loss.backward(); probe_opt.step()
                total   += loss.item() * len(X_b)
                correct += (logits.argmax(1) == y_b).sum().item()
            print(f"    Epoch {ep}/{self.epochs}  loss={total/len(X):.4f}  acc={correct/len(X):.4f}")

    def predict_proba(self, X):
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.vae.eval(); self.probe.eval()
        with torch.no_grad():
            mu, _ = self.vae.encode(X_t)
            logits = self.probe(mu)
        return F.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
