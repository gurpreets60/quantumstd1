"""
CapsNet (Capsule Network) — capsules output pose vectors instead of scalar activations.

QCQ-CNN (Quantum Circuit + Capsule CNN) analog: dynamic routing by agreement
resembles quantum amplitude routing, where interference determines which
higher-level capsule receives each lower-level capsule's vote.

Architecture:
  Conv1 (1->256, k=9) -> 20x20 feature maps
  PrimaryCaps (32 capsules, dim=8) -> 1152 primary capsules (each 8-D vector)
  DigitCaps (num_classes capsules, dim=16) via 3 iterations of dynamic routing
  Output: capsule lengths in [0,1] treated as class probabilities

Squash function: maps capsule vectors so ||v|| in (0,1) while preserving direction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .torch_base import TorchModel


def _squash(x, dim=-1):
    """Squash: normalize vector magnitudes to (0,1) while preserving direction."""
    sq = (x ** 2).sum(dim=dim, keepdim=True)
    return sq / (1 + sq) * x / (sq.sqrt() + 1e-8)


class _PrimaryCaps(nn.Module):
    """Conv layer -> flat set of 8-D primary capsules."""

    def __init__(self, in_ch=256, num_caps=32, cap_dim=8):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim  = cap_dim
        # Output: num_caps*cap_dim channels; stride=2 to reduce spatial dims
        self.conv = nn.Conv2d(in_ch, num_caps * cap_dim, kernel_size=9, stride=2)

    def forward(self, x):
        out = self.conv(x)                         # (N, num_caps*cap_dim, H, W)
        N, _, H, W = out.shape
        out = out.view(N, self.num_caps, self.cap_dim, H * W)
        out = out.permute(0, 1, 3, 2).contiguous()  # (N, num_caps, H*W, cap_dim)
        out = out.view(N, -1, self.cap_dim)          # (N, num_caps*H*W, cap_dim)
        return _squash(out)                          # squash all primary capsules


class _DigitCaps(nn.Module):
    """Dynamic routing by agreement: primary capsules vote for digit capsules."""

    def __init__(self, num_in, num_out, in_dim, out_dim, num_routing=3):
        super().__init__()
        self.num_routing = num_routing
        self.num_out     = num_out
        # W[i, j]: transformation matrix for capsule i -> class j
        self.W = nn.Parameter(torch.randn(1, num_in, num_out, out_dim, in_dim) * 0.01)

    def forward(self, u):
        # u: (N, num_in, in_dim)
        N   = u.shape[0]
        u_  = u.unsqueeze(2).unsqueeze(4)              # (N, num_in, 1, in_dim, 1)
        # Compute votes: u_hat[i,j] = W[i,j] @ u[i]
        u_hat = torch.matmul(self.W, u_).squeeze(-1)   # (N, num_in, num_out, out_dim)

        # Dynamic routing (no gradient through b)
        b = torch.zeros(N, u.shape[1], self.num_out, device=u.device)
        for r in range(self.num_routing):
            c     = F.softmax(b, dim=2).unsqueeze(3)   # coupling coefficients (N, in, out, 1)
            s     = (c * u_hat).sum(dim=1)              # weighted sum of votes (N, out, out_dim)
            v     = _squash(s)                          # squash -> digit capsule output
            if r < self.num_routing - 1:
                # Update routing logits: capsule i agrees with class j if u_hat . v is large
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)
        return v  # (N, num_out, out_dim)


class _CapsNetCore(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1   = nn.Sequential(nn.Conv2d(1, 256, kernel_size=9), nn.ReLU(inplace=True))
        # After conv1 (k=9, stride=1): 28x28 -> 20x20
        self.primary = _PrimaryCaps(in_ch=256, num_caps=32, cap_dim=8)
        # After PrimaryCaps (k=9, stride=2): 20x20 -> 6x6 -> 32*6*6=1152 capsules
        self.digit   = _DigitCaps(num_in=32 * 6 * 6, num_out=num_classes, in_dim=8, out_dim=16)

    def forward(self, x):
        x = self.conv1(x)             # (N, 256, 20, 20)
        u = self.primary(x)           # (N, 1152, 8) primary capsules
        v = self.digit(u)             # (N, num_classes, 16) digit capsules
        return v.norm(dim=-1)         # capsule lengths as class scores (N, num_classes)


class CapsNetModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("CAPSNET", epochs=epochs, img_shape=(1, 28, 28))

    def _build_net(self, num_classes):
        return _CapsNetCore(num_classes)
