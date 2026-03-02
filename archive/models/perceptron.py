"""Perceptron classifier for stock prediction."""
from sklearn.linear_model import Perceptron
from .sklearn_base import SklearnTrainer


class PerceptronTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'PERCEPTRON',
            Perceptron(
                max_iter=200,
                tol=1e-3,
                random_state=42,
            ),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

