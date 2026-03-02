"""
K-Nearest Neighbors with k=3 — standard KNN baseline.

Classifies each test sample by majority vote of its 3 nearest training neighbors
(Euclidean distance in pixel space). Simple, non-parametric, no training phase.
~97% on MNIST with k=3.

Note: prediction is slow (computes distance to all 60k training points per query).
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from .base import BaseModel


class KNN3Model(BaseModel):
    def __init__(self):
        super().__init__("KNN_K3")
        self.model = KNeighborsClassifier(
            n_neighbors=3,    # vote among 3 closest neighbors
            metric="euclidean",
            n_jobs=-1,        # parallelize distance computation
        )

    def train(self, X, y):
        # KNN has no real training — just stores the data
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
