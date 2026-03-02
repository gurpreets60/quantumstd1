"""
PCA + Logistic Regression — dimensionality reduction then linear classification.

PCA projects 784 pixels to 100 principal components (capturing ~90% variance).
LR then classifies in the compressed space — much faster than raw-pixel LR.
Classical analog: PCA compression before a variational circuit (VQC).
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from .base import BaseModel


class PCALogRegModel(BaseModel):
    def __init__(self):
        super().__init__("PCA_LOGREG")
        self.model = Pipeline([
            ("pca", PCA(n_components=100, random_state=42)),  # 784 -> 100 principal components
            ("lr",  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", n_jobs=-1)),
        ])

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
