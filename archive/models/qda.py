"""Quadratic Discriminant Analysis classifier for stock prediction."""
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from .sklearn_base import SklearnTrainer


class QDATrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        # Higher reg_param (0.5) to handle rank-deficient covariance when
        # features > samples per class (55 features, pilot uses 50 samples).
        super().__init__(
            'QDA',
            QuadraticDiscriminantAnalysis(reg_param=0.5),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

