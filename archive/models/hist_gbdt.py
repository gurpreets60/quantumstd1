"""Histogram-based Gradient Boosting classifier (small) for stock prediction."""
from sklearn.ensemble import HistGradientBoostingClassifier
from .sklearn_base import SklearnTrainer


class HistGBDTTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'HIST GBDT',
            HistGradientBoostingClassifier(
                max_iter=30,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            ),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

