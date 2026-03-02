"""
Gaussian Naive Bayes — assumes pixel values follow per-class Gaussian distributions.

P(x_i | class) ~ N(mu_class_i, sigma_class_i^2). Very fast, no iterations.
Naive independence assumption is wrong for pixels, but still ~55-80% on MNIST.
var_smoothing prevents zero-variance pixels (constant across a class) from
causing division-by-zero.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from .base import BaseModel


class GaussianNBModel(BaseModel):
    def __init__(self):
        super().__init__("GAUSSIAN_NB")
        self.model = GaussianNB(
            var_smoothing=1e-9,  # add fraction of max variance to all variances
                                 # prevents log(0) for near-constant pixels
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
