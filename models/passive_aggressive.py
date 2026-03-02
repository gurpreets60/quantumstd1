"""
Passive-Aggressive Classifier — online large-margin learning algorithm.

"Passive": if correctly classified with enough margin, don't update.
"Aggressive": if misclassified or margin too small, update to enforce margin.
Online algorithm (no batch gradient descent). No native predict_proba —
wrapped with CalibratedClassifierCV for probability output.
"""

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.calibration import CalibratedClassifierCV
from .base import BaseModel


class PassiveAggressiveModel(BaseModel):
    def __init__(self):
        super().__init__("PASSIVE_AGGRESSIVE")
        self.model = CalibratedClassifierCV(
            PassiveAggressiveClassifier(
                C=1.0,           # aggressiveness parameter (max step size)
                max_iter=1000,   # passes through the data
                random_state=42,
            ),
            cv=3,
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
