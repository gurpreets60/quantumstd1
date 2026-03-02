"""
Ridge Classifier + Platt Calibration — L2-regularized linear classifier.

RidgeClassifier solves a regression problem (y in {-1, 1} per class) instead
of optimizing log-loss. Faster than LogisticRegression but no native predict_proba.
CalibratedClassifierCV wraps it with Platt scaling (sigmoid fit) to get probabilities.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from .base import BaseModel


class RidgeCalibratedModel(BaseModel):
    def __init__(self):
        super().__init__("RIDGE_CALIBRATED")
        self.model = CalibratedClassifierCV(
            RidgeClassifier(alpha=1.0),   # alpha: L2 regularization strength
            cv=3,                         # 3-fold cross-val for calibration fitting
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
