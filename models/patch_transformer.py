"""
Patch Transformer (Vision Transformer lite) — image as sequence of patches.

Splits 28x28 image into 4x4=16 non-overlapping 7x7 patches.
Each patch is linearly embedded, positional encodings added, then
multi-head self-attention layers process the patch sequence.
A learnable CLS token aggregates global information for classification.
QTL analog: quantum attention over patch-level quantum feature embeddings.
"""

import torch
import torch.nn as nn
from .torch_base import TorchModel


class PatchTransformerNet(nn.Module):
    def __init__(self, num_classes, patch_size=7, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (28 // patch_size) ** 2          # 4x4 = 16 patches
        patch_dim        = patch_size * patch_size           # 49 pixels per patch

        self.embed     = nn.Linear(patch_dim, d_model)       # patch -> d_model embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))    # learnable CLS
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (N, 1, 28, 28)
        N  = x.shape[0]
        ps = self.patch_size
        # Extract non-overlapping patches using unfold
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)     # (N, 1, 4, 4, ps, ps)
        patches = patches.contiguous().view(N, -1, ps * ps)  # (N, 16, 49)
        tokens  = self.embed(patches)                         # (N, 16, d_model)
        cls     = self.cls_token.expand(N, -1, -1)           # (N, 1, d_model)
        tokens  = torch.cat([cls, tokens], dim=1)             # (N, 17, d_model)
        tokens  = tokens + self.pos_embed                     # add learned positional encoding
        out     = self.transformer(tokens)                    # (N, 17, d_model)
        return self.head(out[:, 0])                           # CLS token -> class logits


class PatchTransformerModel(TorchModel):
    def __init__(self, epochs=10):
        super().__init__("PATCH_TRANSFORMER", epochs=epochs, img_shape=(1, 28, 28))

    def _build_net(self, num_classes):
        return PatchTransformerNet(num_classes)
