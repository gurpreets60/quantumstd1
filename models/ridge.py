"""Ridge classifier for stock prediction."""
from sklearn.linear_model import RidgeClassifier
from .sklearn_base import SklearnTrainer


class RidgeTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'RIDGE',
            RidgeClassifier(alpha=1.0),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

