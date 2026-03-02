"""
Bagging (Bootstrap Aggregating) with Decision Trees — Quantum Ensemble analog.

Each tree trained on a random 80% bootstrap sample of the data.
Averaging over diverse trees reduces variance. Unlike RF, no feature subsampling.
Quantum Ensemble analog: multiple quantum circuit instances trained on subsets.
"""

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from .base import BaseModel


class BaggingModel(BaseModel):
    def __init__(self):
        super().__init__("BAGGING")
        self.model = BaggingClassifier(
            estimator=DecisionTreeClassifier(),  # base learner: full decision tree
            n_estimators=50,    # 50 trees (each on a different bootstrap sample)
            max_samples=0.8,    # use 80% of training data per tree
            random_state=42,
            n_jobs=-1,
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
