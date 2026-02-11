"""AdaBoost classifier for stock prediction (small)."""
from sklearn.ensemble import AdaBoostClassifier
from .sklearn_base import SklearnTrainer


class AdaBoostTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'ADABOOST',
            AdaBoostClassifier(
                n_estimators=25,
                learning_rate=0.5,
                random_state=42,
            ),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

