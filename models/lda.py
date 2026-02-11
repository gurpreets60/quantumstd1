"""Linear Discriminant Analysis classifier for stock prediction."""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .sklearn_base import SklearnTrainer


class LDATrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'LDA',
            LinearDiscriminantAnalysis(solver='svd'),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

