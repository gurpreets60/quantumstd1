"""Small Decision Tree classifier for stock prediction."""
from sklearn.tree import DecisionTreeClassifier
from .sklearn_base import SklearnTrainer


class DecisionTreeTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'DECISION TREE',
            DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=10,
                random_state=42,
            ),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

