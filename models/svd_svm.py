"""
Truncated SVD + SVM — LSA-style matrix factorization then classification.

TruncatedSVD (aka LSA) decomposes X into U * S * Vt and keeps top-k singular
vectors. Unlike PCA, it does NOT center the data first — better for sparse inputs.
RBF SVM classifies in the low-rank embedding space.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from .base import BaseModel


class SVDSVMModel(BaseModel):
    def __init__(self):
        super().__init__("SVD_SVM")
        self.model = Pipeline([
            ("svd", TruncatedSVD(n_components=80, random_state=42)),  # keep top 80 singular vectors
            ("svm", SVC(
                kernel="rbf",
                C=5.0,
                gamma="scale",
                probability=True,
                random_state=42,
            )),
        ])

    def train(self, X, y):
        print("  Warning: SVD+SVM may take a few minutes on full MNIST.")
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
