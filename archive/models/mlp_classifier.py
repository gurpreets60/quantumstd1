"""MLP (Multi-layer Perceptron) classifier for stock prediction."""
from sklearn.neural_network import MLPClassifier
from .sklearn_base import SklearnTrainer


class MLPTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'MLP CLASSIFIER',
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=50,
                          random_state=42),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget)
