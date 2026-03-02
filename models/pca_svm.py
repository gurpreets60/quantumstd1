"""
PCA + SVM (RBF) — classic combination, competitive MNIST baseline.

PCA reduces to 64 components; SVM then finds the optimal margin in that space.
SVM in low-dim PCA space is much faster than SVM on raw 784-dim pixels.
Q-Kernel analog: quantum feature map replaces PCA + RBF kernel transformation.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from .base import BaseModel


class PCASVMModel(BaseModel):
    def __init__(self):
        super().__init__("PCA_SVM")
        self.model = Pipeline([
            ("pca", PCA(n_components=64, random_state=42)),  # 784 -> 64 dims
            ("svm", SVC(
                kernel="rbf",
                C=10.0,             # large C: less regularization (tighter margin)
                gamma="scale",      # 1/(n_features * X.var())
                probability=True,
                random_state=42,
            )),
        ])

    def train(self, X, y):
        print("  Warning: PCA+SVM may take a few minutes on full MNIST.")
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
