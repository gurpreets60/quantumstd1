"""
RBM + Logistic Regression — Restricted Boltzmann Machine feature extraction.

RBM is a generative energy-based model. It learns latent features by
maximizing log-likelihood via Contrastive Divergence (CD-1).
The hidden units form a learned feature representation used by LR.
QAE analog: quantum autoencoder compresses quantum states to a latent register,
analogous to RBM learning a compressed binary latent representation.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from .base import BaseModel


class RBMLogRegModel(BaseModel):
    def __init__(self):
        super().__init__("RBM_LOGREG")
        self.model = Pipeline([
            ("rbm", BernoulliRBM(
                n_components=256,    # number of hidden units (latent features)
                learning_rate=0.01,  # CD-1 step size
                n_iter=10,           # training epochs through the data
                random_state=42,
            )),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)),
        ])

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
