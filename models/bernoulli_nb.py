"""
Bernoulli Naive Bayes — binary feature NB, classic for binarized MNIST.

Assumes each pixel is either ON or OFF (binarized at threshold 0.5).
Models P(pixel_i = 1 | class) independently for each pixel and class.
Simple probabilistic baseline; fast training and prediction.
"""

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from .base import BaseModel


class BernoulliNBModel(BaseModel):
    def __init__(self):
        super().__init__("BERNOULLI_NB")
        self.model = BernoulliNB(
            alpha=1.0,      # Laplace smoothing: adds 1 pseudo-count per pixel/class
            binarize=0.5,   # threshold: pixels > 0.5 are treated as 1, else 0
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
