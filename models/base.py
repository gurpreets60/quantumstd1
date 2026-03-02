"""
Abstract base class for all image classification models.

Every model in models/ must subclass BaseModel and implement:
  - train(X_train, y_train)   -- fit the model
  - predict_proba(X)          -- return (N, K) class probability matrix

Models receive flat numpy arrays (N, pixels) as input normalized to [0, 1].
CNN models reshape to image tensors internally; sklearn models use them flat.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract base for all image classifiers in this pipeline.

    The uniform interface lets main.py treat all models identically:
      model.train(X_train, y_train)
      probs = model.predict_proba(X_test)   # (N, K) probabilities
      preds = probs.argmax(axis=1)           # (N,)   class predictions
    """

    def __init__(self, name: str):
        # name is used as the display label and as the filename prefix when saving predictions
        self.name = name

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the model on training data.

        Args:
            X_train: (N, pixels) float32, pixel values in [0, 1]
            y_train: (N,)        int64,   class labels (0 to K-1)
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probability matrix.

        Args:
            X: (N, pixels) float32, pixel values in [0, 1]

        Returns:
            (N, K) float32 where K = number of classes, each row sums to 1.
        """
        ...
