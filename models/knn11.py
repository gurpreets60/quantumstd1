"""
K-Nearest Neighbors with k=11, distance-weighted — smoother KNN variant.

Distance weighting: closer neighbors contribute more to the vote (weight=1/dist).
Odd k=11 avoids ties and is commonly cited in MNIST literature for robustness.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from .base import BaseModel


class KNN11Model(BaseModel):
    def __init__(self):
        super().__init__("KNN_K11_DIST")
        self.model = KNeighborsClassifier(
            n_neighbors=11,
            weights="distance",   # weight votes by 1/distance (closer = more influence)
            metric="euclidean",
            n_jobs=-1,
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
