"""KNN (k=11, distance weights) classifier for stock prediction."""
from sklearn.neighbors import KNeighborsClassifier
from .sklearn_base import SklearnTrainer


class KNN11DistTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'KNN-11-DIST',
            KNeighborsClassifier(n_neighbors=11, weights='distance'),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget
        )

