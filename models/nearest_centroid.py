"""
Nearest Centroid — assigns each sample to the class with the closest mean.

Computes the centroid (mean pixel vector) of each class during training.
At test time, assigns label of the nearest centroid (Euclidean distance).
Extremely fast but ignores within-class variance. Amplitude QNN analog:
quantum state amplitude encoding + nearest quantum state classification.
"""

import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.calibration import CalibratedClassifierCV
from .base import BaseModel


class NearestCentroidModel(BaseModel):
    def __init__(self):
        super().__init__("NEAREST_CENTROID")
        # NearestCentroid has no predict_proba — wrap with Platt scaling calibration
        self.model = CalibratedClassifierCV(NearestCentroid(), cv=3)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
