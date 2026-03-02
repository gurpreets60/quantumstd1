"""LinearSVC classifier for stock prediction."""
from sklearn.svm import LinearSVC
from .sklearn_base import SklearnTrainer


class LinearSVCTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'LINEAR SVC',
            LinearSVC(C=0.5, tol=1e-3, max_iter=2000, random_state=42),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

