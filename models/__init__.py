"""
Model auto-discovery for the image classification pipeline.

Any .py file in models/ that defines a BaseModel subclass is automatically
found and registered here — no manual imports needed. To add a new model,
just create models/your_model.py with a class that extends BaseModel.

Usage:
    from models import get_models
    models = get_models("all", args)   # returns list of BaseModel instances
"""

import pkgutil
import importlib
import inspect
from pathlib import Path
from .base import BaseModel


def _discover_models() -> dict:
    """
    Scan the models/ directory for all BaseModel subclasses.

    Uses pkgutil to iterate modules (same approach as quantumstd1).
    Returns dict {ClassName: class}.
    """
    registry = {}
    models_dir = str(Path(__file__).parent)

    for _, module_name, _ in pkgutil.iter_modules([models_dir]):
        if module_name == "base":
            continue  # skip the abstract base class itself

        # Dynamically import each module in models/
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Find all classes in the module that subclass BaseModel
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseModel) and obj is not BaseModel:
                registry[name] = obj

    return registry


# Maps --model flag values to class name prefixes for filtering
_FLAG_MAP = {
    "rf":  "RandomForest",   # e.g. RandomForestModel
    "cnn": "CNN",            # e.g. CNNModel
    "all": None,             # all discovered models
}


def get_models(model_flag: str, args) -> list:
    """
    Return instantiated model objects based on the --model CLI flag.

    Passes relevant hyperparameters from args (e.g. --epochs for CNN).

    Args:
        model_flag: "rf", "cnn", or "all"
        args:       argparse Namespace (used to pass hyperparams to models)

    Returns:
        List of BaseModel instances, ready to call .train() on.
    """
    registry = _discover_models()

    # Filter registry based on the flag
    if model_flag == "all":
        selected = registry
    elif model_flag in _FLAG_MAP:
        prefix = _FLAG_MAP[model_flag]
        selected = {k: v for k, v in registry.items() if k.startswith(prefix)}
    else:
        # Allow passing a class name directly (e.g. -m RandomForestModel)
        selected = {k: v for k, v in registry.items() if k.lower() == model_flag.lower()}

    if not selected:
        raise ValueError(
            f"No models match '{model_flag}'. "
            f"Available: {list(registry.keys())} or flags: {list(_FLAG_MAP.keys())}"
        )

    # Instantiate each model class, wiring in hyperparams from args
    instances = []
    for name, cls in selected.items():
        if "CNN" in name:
            # Pass epochs from --epochs flag; CNN reads it at construction time
            instances.append(cls(epochs=args.epochs))
        else:
            # sklearn-based models use their own defaults
            instances.append(cls())

    print(f"Models to run: {[m.name for m in instances]}")
    return instances
