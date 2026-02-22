"""Models package — auto-discovers all SklearnTrainer subclasses."""
import importlib
import os
import pkgutil

# Explicit imports for non-sklearn models
from .classical_alstm import AWLSTM
from .quantum_lstm import QuantumTrainer
from .QLSTM_v0_Batch import BatchQLSTMTrainer
from .vqFWP_Batch_time_series import BatchVQFWPTrainer
from .test_oom import OOMTestTrainer

# Auto-discover all SklearnTrainer subclasses from .py files in this directory.
# Each file that defines a SklearnTrainer subclass is imported, and the class
# is registered in SKLEARN_REGISTRY keyed by its model_name.

from .sklearn_base import SklearnTrainer

SKLEARN_REGISTRY = {}  # model_name -> class

_pkg_dir = os.path.dirname(__file__)
_skip = {'__init__', 'sklearn_base', 'monitor', 'cfa', 'classical_alstm',
         'quantum_lstm', 'QLSTM_v0_Batch', 'vqFWP_Batch_time_series',
         'test_oom'}

for _finder, _name, _ispkg in pkgutil.iter_modules([_pkg_dir]):
    if _name in _skip:
        continue
    try:
        _mod = importlib.import_module('.' + _name, __package__)
    except Exception:
        continue
    for _attr in dir(_mod):
        _obj = getattr(_mod, _attr)
        if (isinstance(_obj, type)
                and issubclass(_obj, SklearnTrainer)
                and _obj is not SklearnTrainer):
            if _obj not in SKLEARN_REGISTRY.values():
                SKLEARN_REGISTRY[_attr] = _obj

# Clean up module namespace
del importlib, os, pkgutil, _pkg_dir, _skip
