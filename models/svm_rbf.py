"""
SVM with RBF (Gaussian) kernel — QSVM classical analog.

RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
Maps data into infinite-dimensional feature space, finding a maximum-margin
hyperplane. One of the strongest classical single models on MNIST (~98%).

WARNING: O(n^2) memory, O(n^2-n^3) training time.
On MNIST 60k samples this can take 30-60+ minutes on CPU.
Consider running with a subset or use -m svm_linear for speed.
"""

import numpy as np
from sklearn.svm import SVC
from .base import BaseModel


class SVMRBFModel(BaseModel):
    def __init__(self):
        super().__init__("SVM_RBF")
        self.model = SVC(
            kernel="rbf",
            C=1.0,             # regularization: higher = less regularization
            gamma="scale",     # gamma = 1 / (n_features * X.var()); adapts to data
            probability=True,  # enables predict_proba via Platt scaling (slower)
            random_state=42,
        )

    def train(self, X, y):
        print("  Warning: SVM RBF is slow on large datasets (may take 30+ min on MNIST).")
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
