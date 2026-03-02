"""
AdaBoost — adaptive boosting, reweights samples after each weak learner.

Misclassified samples get higher weight in the next round, forcing the
ensemble to focus on hard examples. Uses decision stumps (depth=1) by default.
Quantum Ensemble analog: adaptive weight update ~ variational parameter update.
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from .base import BaseModel


class AdaBoostModel(BaseModel):
    def __init__(self):
        super().__init__("ADABOOST")
        self.model = AdaBoostClassifier(
            n_estimators=100,    # number of boosting rounds
            learning_rate=0.5,   # shrinks each estimator's contribution
            algorithm="SAMME",   # discrete AdaBoost (required for multiclass predict_proba)
            random_state=42,
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
