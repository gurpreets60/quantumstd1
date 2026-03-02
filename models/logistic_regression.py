"""
Logistic Regression — linear probabilistic classifier, standard MNIST baseline.

Finds a linear decision boundary in pixel space via log-likelihood maximization.
lbfgs solver handles multiclass natively via softmax (one-vs-rest not needed).
~92% on raw MNIST pixels; improves with PCA preprocessing (see pca_logreg).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__("LOGISTIC_REGRESSION")
        self.model = LogisticRegression(
            C=1.0,           # inverse regularization strength (1/lambda)
            max_iter=1000,   # sufficient iterations for convergence on 784-dim data
            solver="lbfgs",  # limited-memory BFGS: efficient for multiclass + dense features
            n_jobs=-1,
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
