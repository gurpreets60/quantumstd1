"""
Data loading for image classification datasets.

Downloads datasets automatically to data_dir/ using torchvision.
Returns flat numpy arrays (N, pixels) normalized to [0, 1], suitable for
both sklearn models (flat) and CNN models (which reshape internally).

Supported datasets (pass --dataset flag to main.py):
    mnist        -- 60k train / 10k test, 10 classes (digits 0-9)
    fashion      -- 60k train / 10k test, 10 classes (clothing items)
    (more datasets will be added: PneumoniaMNIST, DermaMNIST, PathMNIST)
"""

import numpy as np
from pathlib import Path


def load_mnist(data_dir="data"):
    """
    Load MNIST digit classification dataset.

    Downloads to data_dir/MNIST/ via torchvision if not already cached.
    Dataset: 70,000 grayscale images of handwritten digits 0-9, 28x28 pixels.

    Args:
        data_dir: Directory to store the downloaded dataset (default: "data/")

    Returns:
        X_train: (60000, 784) float32, pixel values in [0, 1]
        y_train: (60000,)     int64,   class labels 0-9
        X_test:  (10000, 784) float32
        y_test:  (10000,)     int64
    """
    from torchvision import datasets, transforms

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # ToTensor() converts PIL images (H, W) uint8 to float32 tensors (1, H, W) in [0, 1]
    transform = transforms.ToTensor()

    # download=True fetches from Yann LeCun's servers if not already cached
    train_ds = datasets.MNIST(root=data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # .data is a uint8 tensor (N, 28, 28); flatten to (N, 784) and normalize to [0, 1]
    X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int64)

    X_test  = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test  = test_ds.targets.numpy().astype(np.int64)

    return X_train, y_train, X_test, y_test


def load_fashion_mnist(data_dir="data"):
    """
    Load Fashion-MNIST dataset (drop-in replacement for MNIST).

    Same format as MNIST (28x28 grayscale, 10 classes), but harder:
    classes are clothing items (T-shirt, Trouser, Pullover, ...).

    Args:
        data_dir: Directory to store the downloaded dataset

    Returns:
        X_train, y_train, X_test, y_test (same shapes as load_mnist)
    """
    from torchvision import datasets, transforms

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()

    train_ds = datasets.FashionMNIST(root=data_dir, train=True,  download=True, transform=transform)
    test_ds  = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int64)

    X_test  = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test  = test_ds.targets.numpy().astype(np.int64)

    return X_train, y_train, X_test, y_test


# Map dataset name → loader function (extend here to add new datasets)
DATASET_LOADERS = {
    "mnist":   load_mnist,
    "fashion": load_fashion_mnist,
}


def load_dataset(name, data_dir="data"):
    """
    Dispatch to the correct dataset loader by name.

    Args:
        name:     Dataset name string (e.g. "mnist", "fashion")
        data_dir: Directory to cache downloads

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name](data_dir)
