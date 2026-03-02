"""Dummy baseline (stratified) for stock prediction."""
from sklearn.dummy import DummyClassifier
from .sklearn_base import SklearnTrainer


class DummyStratifiedTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'DUMMY STRATIFIED',
            DummyClassifier(strategy='stratified', random_state=42),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

