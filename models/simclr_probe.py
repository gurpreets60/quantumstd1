"""
SimCLR + Linear Probe — contrastive self-supervised learning.

QSimCLR analog: quantum feature map for contrastive representation learning.
Trains an encoder so that two augmented views of the same image are similar
in latent space (NT-Xent loss), while different images are pushed apart.

NT-Xent loss (normalized temperature-scaled cross entropy):
  For batch of N images: create 2N views via augmentation.
  Loss = pull positive pairs together, push all negatives apart.

Stage 1: contrastive pretraining (no labels used).
Stage 2: linear probe on frozen encoder (uses labels).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .torch_base import TorchModel


class _SimCLREncoder(nn.Module):
    def __init__(self, proj_dim=64):
        super().__init__()
        # Backbone: shared feature extractor
        self.backbone = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        # Projection head: used during contrastive training only (discarded for probing)
        self.projector = nn.Sequential(nn.Linear(128, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, x):
        return self.backbone(x)        # backbone features (used at test time)

    def project(self, x):
        return self.projector(self.backbone(x))   # projected features (used in NT-Xent)


def _nt_xent(z1, z2, temp=0.5):
    """NT-Xent contrastive loss for a batch of pairs (z1[i], z2[i])."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N  = z1.shape[0]
    z  = torch.cat([z1, z2], dim=0)                              # (2N, dim)
    sim = torch.mm(z, z.T) / temp                                 # (2N, 2N) similarity
    # Mask out self-similarity on diagonal
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)


def _augment(X):
    """Light augmentation: Gaussian noise (works on flat vectors without image ops)."""
    return X + 0.1 * torch.randn_like(X)


class SimCLRProbeModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("SIMCLR_PROBE", epochs=epochs, img_shape=(784,))
        self.encoder = self.probe = None

    def _build_net(self, num_classes):
        self.encoder = _SimCLREncoder()
        self.probe   = nn.Linear(128, num_classes)   # probe on backbone output (dim=128)
        return self.encoder

    def train(self, X, y):
        num_classes = len(np.unique(y))
        self._build_net(num_classes)
        self.encoder.to(self.device); self.probe.to(self.device)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dl  = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        # Stage 1: contrastive pretraining (no labels)
        opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        print("  Stage 1: SimCLR contrastive pretraining")
        for ep in range(1, self.epochs + 1):
            total = 0.0
            for X_b, _ in dl:
                X_b = X_b.to(self.device)
                z1   = self.encoder.project(_augment(X_b))   # view 1
                z2   = self.encoder.project(_augment(X_b))   # view 2 (different noise)
                loss = _nt_xent(z1, z2)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(X_b)
            print(f"    Epoch {ep}/{self.epochs}  contrast_loss={total/len(X):.4f}")

        # Stage 2: linear probe on frozen backbone
        self.encoder.eval()
        probe_opt = torch.optim.Adam(self.probe.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        print("  Stage 2: linear probe on frozen backbone")
        for ep in range(1, self.epochs + 1):
            total, correct = 0.0, 0
            for X_b, y_b in dl:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                probe_opt.zero_grad()
                with torch.no_grad():
                    h = self.encoder(X_b)
                logits  = self.probe(h)
                loss    = ce(logits, y_b)
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
