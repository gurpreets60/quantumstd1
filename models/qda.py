"""
Quadratic Discriminant Analysis — like LDA but with class-specific covariances.

Each class gets its own covariance matrix → quadratic decision boundaries.
More flexible than LDA but requires regularization on high-dim data to avoid
singular class covariance matrices (many MNIST pixels are near-constant).
"""

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from .base import BaseModel


class QDAModel(BaseModel):
    def __init__(self):
        super().__init__("QDA")
        self.model = QuadraticDiscriminantAnalysis(
            reg_param=0.1,   # shrink class covariances toward identity
                             # avoids singular matrices from near-zero variance pixels
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
