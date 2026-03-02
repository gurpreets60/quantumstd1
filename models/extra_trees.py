"""
Extra Trees (Extremely Randomized Trees) — more random than Random Forest.

Random Forest: random subset of features, optimal split threshold.
Extra Trees:   random subset of features, RANDOM split threshold (faster, more regularized).
Randomizing thresholds reduces variance at cost of slight bias increase.
Usually slightly faster than RF with similar accuracy (~97% MNIST).
"""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from .base import BaseModel


class ExtraTreesModel(BaseModel):
    def __init__(self):
        super().__init__("EXTRA_TREES")
        self.model = ExtraTreesClassifier(
            n_estimators=100,   # number of trees
            random_state=42,
            n_jobs=-1,          # parallelize across all CPU cores
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
