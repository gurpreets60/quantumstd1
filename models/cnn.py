"""
Convolutional Neural Network (CNN) for image classification.

Architecture (for 28x28 grayscale MNIST input):
  Block 1: Conv2d(1→32, 3x3, pad=1) → ReLU → MaxPool(2x2)   [28x28 → 14x14]
  Block 2: Conv2d(32→64, 3x3, pad=1) → ReLU → MaxPool(2x2)  [14x14 → 7x7]
  Head:    Flatten(3136) → Linear(128) → ReLU → Dropout(0.25) → Linear(10)

Why CNNs work for images:
  - Conv layers learn local spatial patterns (edges, curves) shared across pixels
  - MaxPool downsamples, providing spatial invariance and reducing parameters
  - Dropout prevents co-adaptation of neurons (regularization)

Training:
  - Optimizer: Adam (adaptive learning rate, fast convergence)
  - Loss: CrossEntropyLoss (combines LogSoftmax + NLLLoss, numerically stable)
  - ~99% test accuracy on MNIST in 5 epochs

Hyperparameters (all documented below):
  epochs       -- training iterations over the full dataset
  batch_size   -- samples per gradient update
  lr           -- Adam learning rate
  num_classes  -- output dimension (10 for MNIST)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from .base import BaseModel


# ---------------------------------------------------------------------------
# Neural network definition
# ---------------------------------------------------------------------------

class _CNN(nn.Module):
    """Two-block CNN for 28x28 grayscale images. Not used directly — see CNNModel."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Block 1: detect low-level features (edges, corners)
        # 1 input channel (grayscale), 32 feature maps, 3x3 kernel, padding=1 keeps spatial size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # Block 2: detect higher-level patterns (curves, shapes)
        # 32 input channels → 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # MaxPool halves spatial size each time: 28x28 → 14x14 → 7x7
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected: 64 channels × 7 × 7 spatial = 3136 units → 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)

        # Dropout: randomly zero 25% of activations during training to prevent overfitting
        self.dropout = nn.Dropout(p=0.25)

        # Output: 128 → num_classes raw logits (softmax applied separately in predict_proba)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))   # (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (N, 64, 7, 7)
        x = x.flatten(start_dim=1)             # (N, 3136)
        x = F.relu(self.fc1(x))               # (N, 128)
        x = self.dropout(x)                    # (N, 128) with 25% zeroed during train
        return self.fc2(x)                     # (N, num_classes) — raw logits


# ---------------------------------------------------------------------------
# BaseModel wrapper
# ---------------------------------------------------------------------------

class CNNModel(BaseModel):
    """
    CNN trainer that accepts flat numpy arrays (N, 784) and reshapes internally.
    Follows the BaseModel interface so main.py can treat it like any other model.
    """

    def __init__(
        self,
        num_classes: int = 10,
        epochs: int = 5,
        batch_size: int = 256,
        lr: float = 1e-3,
    ):
        """
        Args:
            num_classes: Output dimension; 10 for MNIST (digits 0-9).

            epochs: Full passes over the training data.
                5 epochs achieves ~99% on MNIST. Beyond 10 gives diminishing returns
                and risks overfitting on smaller datasets.

            batch_size: Samples per gradient update.
                256 balances GPU utilization and gradient stability.
                Larger batches train faster but may generalize slightly worse.

            lr: Learning rate for Adam optimizer.
                1e-3 is Adam's default and works well for MNIST.
                Reduce to 1e-4 if loss is unstable or oscillating.
        """
        super().__init__("CNN")

        self.num_classes = num_classes
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr

        # Use GPU if available (CUDA), otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  CNN device: {self.device}")

        # Instantiate the network and move all parameters to the selected device
        self.net = _CNN(num_classes).to(self.device)

    def _to_loader(self, X: np.ndarray, y: np.ndarray = None, shuffle: bool = False) -> DataLoader:
        """
        Convert flat numpy arrays to a PyTorch DataLoader.

        Reshapes (N, 784) → (N, 1, 28, 28) for convolutional layers.
        """
        # Reshape flat pixels into (N, channels=1, H=28, W=28)
        X_t = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 28, 28)

        if y is not None:
            y_t = torch.tensor(y, dtype=torch.long)
            ds = TensorDataset(X_t, y_t)
        else:
            ds = TensorDataset(X_t)

        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the CNN using Adam optimizer and cross-entropy loss.

        Prints loss and accuracy after each epoch so progress is visible.
        """
        loader    = self._to_loader(X_train, y_train, shuffle=True)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn   = nn.CrossEntropyLoss()  # numerically stable log-softmax + NLL

        self.net.train()  # enable dropout during training

        for epoch in range(1, self.epochs + 1):
            total_loss, correct = 0.0, 0

            for X_batch, y_batch in loader:
                # Move data to GPU/CPU
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()           # clear gradients from previous step
                logits = self.net(X_batch)      # forward pass → raw logits (N, 10)
                loss   = loss_fn(logits, y_batch)  # cross-entropy loss
                loss.backward()                 # compute gradients
                optimizer.step()                # update weights

                total_loss += loss.item() * len(X_batch)
                correct    += (logits.argmax(dim=1) == y_batch).sum().item()

            avg_loss = total_loss / len(X_train)
            acc      = correct   / len(X_train)
            print(f"  Epoch {epoch}/{self.epochs}  loss={avg_loss:.4f}  train_acc={acc:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Run forward pass and return softmax class probabilities.

        Disables gradient computation (torch.no_grad) and dropout (eval mode)
        during inference — both reduce memory and improve speed.

        Returns: (N, K) float32 where each row sums to 1.
        """
        loader = self._to_loader(X, shuffle=False)
        self.net.eval()  # disable dropout during inference
        probs_list = []

        with torch.no_grad():  # no gradients needed → faster, less memory
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                logits  = self.net(X_batch)
                # softmax converts raw logits to probabilities summing to 1
                probs   = F.softmax(logits, dim=1)
                probs_list.append(probs.cpu().numpy())

        return np.concatenate(probs_list, axis=0).astype(np.float32)
