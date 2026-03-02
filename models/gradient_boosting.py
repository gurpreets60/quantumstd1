"""
Histogram Gradient Boosting — fast GBM using binned (bucketed) features.

sklearn's HistGradientBoostingClassifier bins pixel values into 256 buckets,
enabling much faster split-finding than classic GBM. ~97% on MNIST.
Quantum Ensemble analog: ensemble of weak learners with adaptive weighting.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from .base import BaseModel


class GradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__("HIST_GBM")
        self.model = HistGradientBoostingClassifier(
            max_iter=200,        # number of boosting rounds (trees)
            max_depth=6,         # max depth per tree
            learning_rate=0.1,   # shrinkage: scales each tree's contribution
            random_state=42,
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
