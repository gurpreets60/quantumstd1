"""
Linear Discriminant Analysis — finds axes that maximize between-class separation.

Projects data to at most (K-1) dimensions where classes are most separable.
Assumes Gaussian class conditionals with shared covariance matrix.
Also useful as a dimensionality reducer (784 -> 9 dims for MNIST).
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .base import BaseModel


class LDAModel(BaseModel):
    def __init__(self):
        super().__init__("LDA")
        self.model = LinearDiscriminantAnalysis(
            solver="svd",    # SVD avoids computing the covariance matrix explicitly
                             # (numerically stable on high-dimensional inputs)
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
