"""
K-Nearest Neighbors with k=5 — standard KNN baseline (more robust than k=3).

More neighbors = smoother decision boundary, less sensitive to noise.
Usually slightly more robust than k=3 on noisy datasets.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from .base import BaseModel


class KNN5Model(BaseModel):
    def __init__(self):
        super().__init__("KNN_K5")
        self.model = KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean",
            n_jobs=-1,
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
