"""
Decision Tree — axis-aligned recursive binary splits on pixel values.

Simple interpretable baseline. Without depth limit, it memorizes training data
(~100% train acc). Test accuracy on MNIST ~88%. Useful as a weak learner base
for ensemble methods (AdaBoost, Bagging, etc.).
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .base import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__("DECISION_TREE")
        self.model = DecisionTreeClassifier(
            random_state=42,    # reproducible splits
            # max_depth=None: grow full tree (pure leaves) — intentional for baseline
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X).astype(np.float32)
