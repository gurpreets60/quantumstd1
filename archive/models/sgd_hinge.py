"""SGD (linear SVM hinge) classifier for stock prediction."""
from sklearn.linear_model import SGDClassifier
from .sklearn_base import SklearnTrainer


class SGDHingeTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'SGD HINGE',
            SGDClassifier(
                loss='hinge',
                alpha=1e-4,
                max_iter=200,
                tol=1e-3,
                random_state=42,
            ),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

