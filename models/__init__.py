"""
Model auto-discovery for the image classification pipeline.

Any .py file in models/ that defines a BaseModel subclass is automatically
found and registered — no manual imports needed.

Model flags (-m):
  all      -- every discovered model
  sklearn  -- all non-PyTorch models (classical ML + pipelines)
  deep     -- all TorchModel subclasses (PyTorch deep learning)
  rf       -- Random Forest
  cnn      -- original CNN
  svm      -- all SVM variants
  knn      -- all KNN variants
  mlp      -- all MLP variants
  <name>   -- any case-insensitive prefix of a class name
"""

import pkgutil
import importlib
import inspect
from pathlib import Path
from .base import BaseModel

# Modules that define abstract bases, not concrete models
_SKIP = {"base", "torch_base"}


def _discover_models() -> dict:
    """Scan models/ for all concrete BaseModel subclasses. Returns {ClassName: class}."""
    registry   = {}
    models_dir = str(Path(__file__).parent)

    for _, module_name, _ in pkgutil.iter_modules([models_dir]):
        if module_name in _SKIP:
            continue
        module = importlib.import_module(f".{module_name}", package=__name__)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseModel) and obj is not BaseModel:
                registry[name] = obj

    return registry


def _torch_base():
    """Return TorchModel class (or None if not yet created)."""
    try:
        from .torch_base import TorchModel
        return TorchModel
    except ImportError:
        return None


def get_models(model_flag: str, args) -> list:
    """
    Return instantiated models based on the --model flag.

    Passes epochs= to any model whose __init__ accepts it (all TorchModel subclasses).
    Uses case-insensitive prefix matching so -m svm matches SVMRBFModel, etc.
    """
    registry   = _discover_models()
    TorchModel = _torch_base()
    flag       = model_flag.lower()

    # Category flags
    if flag == "all":
        selected = registry

    elif flag == "sklearn":
        # Non-PyTorch models only
        selected = {k: v for k, v in registry.items()
                    if TorchModel is None or not issubclass(v, TorchModel)}

    elif flag == "deep":
        # PyTorch TorchModel subclasses only
        selected = {k: v for k, v in registry.items()
                    if TorchModel is not None and issubclass(v, TorchModel)}

    else:
        # Case-insensitive prefix match: -m svm -> SVMRBFModel, SVMLinearModel, ...
        selected = {k: v for k, v in registry.items()
                    if k.lower().startswith(flag)}

    if not selected:
        raise ValueError(
            f"No models match '{model_flag}'.\n"
            f"Flags: all, sklearn, deep\n"
            f"Prefixes: rf, cnn, svm, knn, mlp, lda, qda, pca, rbm, fft, "
            f"lenet, resnet, densenet, mobilenet, lstm, patch, autoencoder, "
            f"vae, simclr, randomfilter, capsnet, ...\n"
            f"All classes: {sorted(registry.keys())}"
        )

    # Instantiate: pass epochs= only if the constructor accepts it
    instances = []
    for name, cls in sorted(selected.items()):
        sig    = inspect.signature(cls.__init__)
        kwargs = {"epochs": args.epochs} if "epochs" in sig.parameters else {}
        instances.append(cls(**kwargs))

    print(f"Models to run ({len(instances)}): {[m.name for m in instances]}")
    return instances
