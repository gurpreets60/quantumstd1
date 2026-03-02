"""
SVM with linear kernel — linear QSVM classical analog.

Finds a maximum-margin linear hyperplane in the original pixel space.
Faster than RBF SVM but less expressive. Uses calibration for probabilities
since LinearSVC doesn't expose predict_proba natively.
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from .base import BaseModel


class SVMLinearModel(BaseModel):
    def __init__(self):
        super().__init__("SVM_LINEAR")
        # LinearSVC is faster than SVC(kernel='linear') for large datasets
        # CalibratedClassifierCV adds Platt scaling to get probabilities
        self.model = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000, random_state=42),
            cv=3,
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
