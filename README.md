# quantumstd1

Quantum and classical models for stock movement prediction. Runs up to 6 models
(Random Forest, Gradient Boosting, MLP, Logistic Regression, Quantum LSTM,
Classical ALSTM), then fuses their predictions with Combinatorial Fusion
Analysis (CFA). Each model is guarded by per-model RAM and time limits and
monitored with live system stats (RAM, CPU, GPU, ETA).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13 (TensorFlow does not yet support 3.14+).

```bash
# or run: bash setup.sh
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python tensorflow tf-keras scikit-learn scipy numpy psutil rich pennylane torch pandas
```

## Quick Start

```bash
# all sklearn models (fast, <2s total)
uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1

# single model
uv run --python .venv/bin/python pred_lstm.py -o train -m rf

# full pipeline: sklearn + quantum (1 epoch) + classical (1 epoch)
uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1 -qe 1 -e 1 --time_limit 30

# run CFA on previous results (instant, no training)
uv run --python .venv/bin/python pred_lstm.py -o cfa
```

### Output

Each run creates a timestamped folder under `data/`:

```
data/run_YYYYMMDD_HHMMSS/
  results.csv              # per-model accuracy, MCC, timing
  cfa_results.csv          # CFA fusion scores (all combinations x methods)
  pred_<MODEL>_val.csv     # per-model validation predictions
  pred_<MODEL>_test.csv    # per-model test predictions
```

## Actions

| Flag | Action |
|------|--------|
| `-o train` | Train model(s) and run CFA |
| `-o cfa` | Run CFA on a previous run (no training, instant) |
| `-o test` | Evaluate on val/test sets |
| `-o pred` | Save predictions to file |
| `-o adv` | Evaluate adversarial robustness |
| `-o latent` | Extract latent representations |

## Model Selection

Use `-m` / `--model` to run a single model, or `--sklearn 1` for all sklearn models:

| `-m` value | Model |
|------------|-------|
| `all` | All enabled models (default) |
| `rf` | Random Forest |
| `gb` | Gradient Boosting |
| `mlp` | MLP Classifier |
| `lr` | Logistic Regression |
| `quantum` | Quantum LSTM |
| `classical` | Classical ALSTM |
| `oom` | OOM Test |

```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m rf
uv run --python .venv/bin/python pred_lstm.py -o train -m quantum -qt 15 --time_limit 30
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -e 10 --time_limit 0
```

## CFA (Combinatorial Fusion Analysis)

CFA fuses predictions from multiple models. Runs automatically after training,
or standalone with `-o cfa`:

```bash
# CFA on most recent run (auto-detected)
uv run --python .venv/bin/python pred_lstm.py -o cfa

# CFA on a specific run
uv run --python .venv/bin/python pred_lstm.py -o cfa --run_dir data/run_20260211_000403

# CFA with selected models only
uv run --python .venv/bin/python pred_lstm.py -o cfa --cfa_models "RANDOM FOREST,GRADIENT BOOST"

# CFA with longer time limit (for many models)
uv run --python .venv/bin/python pred_lstm.py -o cfa --cfa_time 5
```

| Flag | Description | Default |
|------|-------------|---------|
| `--cfa_time` | Time limit for CFA in seconds (0=unlimited) | `1.0` |
| `--cfa_models` | Comma-separated model names to include | all |
| `--run_dir` | Path to a previous run directory | most recent |

CFA evaluates combinations smallest-first and stops when the time limit is
reached. Results are saved to `cfa_results.csv` in the run folder and top-15
are printed to the terminal.

## Guard Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mem_limit` | Per-model RAM limit in GB (0=off) | `1.0` |
| `--time_limit` | Per-model time limit in seconds (0=off) | `1.0` |
| `--test_oom` | Run OOM test model to verify MemoryGuard (0/1) | `0` |
| `--oom_gb` | GB the OOM test tries to allocate | `2.0` |
| `-t` | Hard timeout — kill entire script after N seconds (0=off) | `0` |

Models that exceed their limit are killed and excluded from CFA.

**Note:** The default `--time_limit 1` is strict. Use `--time_limit 30` for
quantum/classical models that need more time, or `--time_limit 0` to disable.

## Model-Specific Options

### Sklearn Models

| Flag | Description | Default |
|------|-------------|---------|
| `--sklearn` | Run all sklearn models: RF, GB, MLP, LR (0/1) | `0` |

Auto-calibrates training set size via pilot fit to stay within time budget.

### Classical ALSTM

| Flag | Description | Default |
|------|-------------|---------|
| `-e` | Number of epochs | `150` |
| `-b` | Batch size | `1024` |
| `-l` | Sequence length | `5` |
| `-u` | LSTM hidden units | `32` |
| `-r` | Learning rate | `0.01` |
| `-a` | Use attention (0/1) | `1` |
| `-v` | Adversarial training (0/1) | `0` |
| `-hi` | Use hinge loss (0/1) | `1` |
| `-f` | Fixed random seed (0/1) | `0` |
| `-g` | Use GPU (0/1) | `0` |
| `-rl` | Reload saved model (0/1) | `0` |
| `-p` | Path to price data | `./data/stocknet-dataset/price/ourpped` |
| `-q` | Path to load model from | `./data/saved_model/acl18_alstm/exp` |
| `-qs` | Path to save model to | `./data/tmp/model` |

### Quantum LSTM

| Flag | Description | Default |
|------|-------------|---------|
| `-qe` | Quantum LSTM epochs (0 to skip) | `0` |
| `-qi` | Compressed input dimension | `2` |
| `-qh` | Quantum hidden size | `2` |
| `-qd` | VQC circuit depth | `1` |
| `-qt` | Time budget in seconds | `10` |

Auto-calibrates by timing a pilot batch, then subsamples data to fit within
the time budget.

## Examples

```bash
# Full pipeline: all sklearn + quantum (3 epochs) + classical (10 epochs)
uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1 -qe 3 -e 10 -qt 15 --time_limit 30

# Classical only, 10 epochs with adversarial training
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -e 10 -v 1 --time_limit 0

# Sklearn models only
uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1

# Minimal quantum test (2 qubits, fastest possible)
uv run --python .venv/bin/python pred_lstm.py -o train -m quantum -qe 1 -qi 1 -qh 1 -qd 1 -qt 5

# Debug run (1 epoch each, all models)
uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1 -qe 1 -e 1 --time_limit 30

# CFA on previous results
uv run --python .venv/bin/python pred_lstm.py -o cfa
```

## Hyperparameter Recipes

### ACL18 dataset

**LSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -a 0 -l 10 -u 32 -l2 10 -f 1 --time_limit 0
```

**ALSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -l 5 -u 4 -l2 1 -f 1 --time_limit 0
```

**Adv-ALSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -l 5 -u 4 -l2 1 -v 1 -rl 1 -q ./data/saved_model/acl18_alstm/exp -la 0.01 -le 0.05 --time_limit 0
```

### KDD17 dataset

**LSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -p ./data/kdd17/ourpped/ -l 5 -u 4 -l2 0.001 -a 0 -f 1 --time_limit 0
```

**ALSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -p ./data/kdd17/ourpped/ -l 15 -u 16 -l2 0.001 -f 1 --time_limit 0
```

**Adv-ALSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -p ./data/kdd17/ourpped/ -l 15 -u 16 -l2 0.001 -v 1 -rl 1 -q ./data/saved_model/kdd17_alstm/model -la 0.05 -le 0.001 -f 1 --time_limit 0
```

## How to Add a New Model

Every model must finish within the **1 GB RAM** and **1 second time** limits or
it gets killed. The `SklearnTrainer` base class handles most of the work — it
auto-calibrates the training set size by timing a pilot fit.

### Step-by-step

**1. Create `models/your_model.py`**

```python
"""Your Model classifier for stock prediction."""
from sklearn.your_module import YourClassifier
from .sklearn_base import SklearnTrainer


class YourModelTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'YOUR MODEL',                          # display name
            YourClassifier(param=value),            # sklearn estimator
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget)
```

**2. Export it in `models/__init__.py`**

```python
from .your_model import YourModelTrainer
```

**3. Wire it into `pred_lstm.py`** — 4 edits:

```python
# (a) Import
from models import (..., YourModelTrainer)

# (b) Add to _MODEL_MAP (pick a short key)
_MODEL_MAP = {
    ...,
    'ym': 'YOUR MODEL',
}

# (c) Add key to the sklearn key list
if key in ('rf', 'gb', 'mlp', 'lr', 'ym'):

# (d) Add to the sklearn registration list and run loop
for sk_name in ['RANDOM FOREST', ..., 'YOUR MODEL']:
...
for name, cls in [('RANDOM FOREST', RandomForestTrainer),
                   ...,
                   ('YOUR MODEL', YourModelTrainer)]:
```

**4. Test it**

```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m ym --time_limit 1 --mem_limit 1
```

### Fitting within 1s / 1GB limits

The `SklearnTrainer` base class automatically subsamples training data to fit
the time budget. But if your model is still too slow or uses too much RAM:

| Problem | Fix |
|---------|-----|
| Killed by TimeGuard | Reduce complexity: fewer estimators (`n_estimators=20`), shallower trees (`max_depth=3`), fewer iterations (`max_iter=50`) |
| Killed by MemoryGuard | Use lighter model, reduce `n_estimators`, set `max_depth`, avoid algorithms that build large internal structures (e.g. KNN with large k) |
| Pilot fit alone takes >0.5s | The base class only uses 50 samples for pilot — if that's too slow, simplify the estimator or use a faster solver |
| MLP convergence warning | Lower `max_iter` or use a simpler architecture (`hidden_layer_sizes=(16,)`) |

**Rule of thumb**: start with minimal parameters, test with `-m ym --time_limit 1`,
and increase complexity only while it still passes. The auto-calibration will
subsample training data as needed, so low `n_estimators` / `max_depth` /
`max_iter` is the main lever.

### Example: Logistic Regression (simplest possible model)

`models/logistic_regression.py` — 8 lines total:

```python
from sklearn.linear_model import LogisticRegression
from .sklearn_base import SklearnTrainer

class LogisticRegressionTrainer(SklearnTrainer):
    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hinge=True, time_budget=0.8):
        super().__init__(
            'LOGISTIC REGRESSION',
            LogisticRegression(max_iter=100, solver='lbfgs', random_state=42),
            tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
            hinge=hinge, time_budget=time_budget)
```

Runs in 0.2s, 0 MB above baseline — well within limits.
