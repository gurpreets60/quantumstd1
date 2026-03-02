"""Bagging of small decision trees (tiny) for stock prediction."""
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from .sklearn_base import SklearnTrainer


class BaggingDTTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        base = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=10,
            random_state=42,
        )
        super().__init__(
            'BAGGING DT2',
            BaggingClassifier(
                estimator=base,
                n_estimators=10,
                max_samples=0.7,
                n_jobs=1,
                random_state=42,
            ),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

