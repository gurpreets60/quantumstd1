"""Gaussian Naive Bayes classifier for stock prediction."""
from sklearn.naive_bayes import GaussianNB
from .sklearn_base import SklearnTrainer


class GaussianNBTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'GAUSSIAN NB',
            GaussianNB(var_smoothing=1e-9),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

