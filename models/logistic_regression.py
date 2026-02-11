"""Logistic Regression classifier for stock prediction."""
from sklearn.linear_model import LogisticRegression
from .sklearn_base import SklearnTrainer


class LogisticRegressionTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'LOGISTIC REGRESSION',
            LogisticRegression(max_iter=100, solver='lbfgs', random_state=42),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget)
