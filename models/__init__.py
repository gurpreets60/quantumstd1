from .classical_alstm import AWLSTM
from .quantum_lstm import QuantumTrainer
from .test_oom import OOMTestTrainer
from .random_forest import RandomForestTrainer
from .gradient_boosting import GradientBoostTrainer
from .mlp_classifier import MLPTrainer
from .logistic_regression import LogisticRegressionTrainer

from .sgd_log import SGDLogTrainer
from .sgd_hinge import SGDHingeTrainer
from .passive_aggressive import PassiveAggressiveTrainer
from .ridge import RidgeTrainer
from .perceptron import PerceptronTrainer
from .lda import LDATrainer
from .qda import QDATrainer
from .gaussian_nb import GaussianNBTrainer
from .nearest_centroid import NearestCentroidTrainer
from .decision_tree import DecisionTreeTrainer
from .extra_trees import ExtraTreesTrainer
from .adaboost import AdaBoostTrainer


from .linear_svc import LinearSVCTrainer
from .hist_gbdt import HistGBDTTrainer
from .bagging_dt import BaggingDTTrainer
from .dummy_mostfreq import DummyMostFreqTrainer
from .dummy_stratified import DummyStratifiedTrainer
from .knn3 import KNN3Trainer
from .knn11_dist import KNN11DistTrainer
from .sgd_modhuber import SGDModHuberTrainer
from .gaussian_nb_smooth8 import GaussianNB1e8Trainer
from .gaussian_nb_smooth7 import GaussianNB1e7Trainer



