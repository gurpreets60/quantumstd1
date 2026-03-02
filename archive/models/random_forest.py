"""Random Forest classifier for stock prediction."""
from sklearn.ensemble import RandomForestClassifier
from .sklearn_base import SklearnTrainer


class RandomForestTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'RANDOM FOREST',
            RandomForestClassifier(n_estimators=20, n_jobs=-1,
                                   random_state=42),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget)
