"""
Random Forest classifier for image classification.

Random Forest builds an ensemble of decision trees, each trained on a
bootstrap sample of the data and a random subset of features at each split.
Predictions are the average (soft vote) across all trees.

Strengths on MNIST:
  - No feature scaling needed (pixel values already in [0, 1])
  - Handles high-dimensional input (784 pixels) without PCA
  - Fast training, deterministic results with fixed random_state
  - ~97% test accuracy on MNIST with 100 trees

Hyperparameters (all documented below):
  n_estimators  -- number of trees (more = better, but slower)
  max_depth     -- max splits per tree (None = grow until pure leaves)
  max_features  -- features considered at each split ("sqrt" ≈ 28 of 784)
  random_state  -- seed for reproducibility
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest wrapper following the BaseModel interface."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,   #Setting a max depth here because it seems like the RF is memorizing/overfitting here.
        max_features: str = "sqrt",
        random_state: int = 42,
    ):
        """
        Args:
            n_estimators: Number of decision trees.
                More trees reduce variance but increase train time linearly.
                100 gives ~97% on MNIST; 200+ improves by <0.1%.

            max_depth: Maximum depth of each tree.
                None = grow until all leaves are pure (or min_samples splits).
                Unlimited depth works well for MNIST; set to 20 to speed up.

            max_features: Number of features to consider at each split.
                "sqrt" = sqrt(784) ≈ 28 features per split (scikit-learn default).
                Reduces correlation between trees, improving ensemble diversity.

            random_state: Random seed for reproducibility.
                Controls bootstrap sampling and feature selection.
        """
        super().__init__("RANDOM_FOREST")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,   # 100 trees in the forest
            max_depth=max_depth,         # grow fully (no depth limit)
            max_features=max_features,   # sqrt(784)≈28 features per split
            random_state=random_state,   # reproducible results
            n_jobs=-1,                   # use all available CPU cores
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the forest on flattened pixel data."""
        print(f"  Fitting RF ({self.model.n_estimators} trees) on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities: fraction of trees voting for each class.

        Internally, each tree votes for one class; predict_proba returns
        the vote fraction across all trees → soft probability.

        Returns: (N, 10) for MNIST (10 digit classes)
        """
        return self.model.predict_proba(X).astype(np.float32)
