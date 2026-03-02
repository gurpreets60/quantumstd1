"""Extra Trees classifier for stock prediction (tiny ensemble)."""
from sklearn.ensemble import ExtraTreesClassifier
from .sklearn_base import SklearnTrainer


class ExtraTreesTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'EXTRA TREES',
            ExtraTreesClassifier(
                n_estimators=10,
                max_depth=5,
                min_samples_leaf=5,
                n_jobs=1,
                random_state=42,
            ),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

