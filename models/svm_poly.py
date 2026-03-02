"""
SVM with polynomial kernel — Q-Kernel classical analog.

Polynomial kernel: K(x,y) = (gamma * x·y + coef0)^degree
Captures interactions between features up to a given degree.
Degree 3 is standard for MNIST (cubic interactions between pixels).

WARNING: Slow on 60k samples like RBF SVM.
"""

import numpy as np
from sklearn.svm import SVC
from .base import BaseModel


class SVMPolyModel(BaseModel):
    def __init__(self):
        super().__init__("SVM_POLY")
        self.model = SVC(
            kernel="poly",
            degree=3,          # cubic polynomial interactions
            C=1.0,
            gamma="scale",
            coef0=1.0,         # bias term in kernel: (gamma*x·y + coef0)^degree
            probability=True,
            random_state=42,
        )

    def train(self, X, y):
        print("  Warning: SVM Poly is slow on large datasets (may take 30+ min on MNIST).")
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
