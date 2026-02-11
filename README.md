# quantumstd1

Quantum and classical models for stock movement prediction. Runs up to 5 models
(Random Forest, Gradient Boosting, MLP, Quantum LSTM, Classical ALSTM), then
fuses their predictions with Combinatorial Fusion Analysis (CFA). Each model is
guarded by per-model RAM and time limits and monitored with live system stats
(RAM, CPU, GPU, ETA).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13 (TensorFlow does not yet support 3.14+).

```bash
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python tensorflow tf-keras scikit-learn scipy numpy psutil rich pennylane torch pandas
```

## Run

```bash
uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1 -qe 5 -e 10
```

This trains the 3 sklearn models, the Quantum LSTM for 5 epochs, then the
Classical ALSTM for 10 epochs, and finishes with CFA fusion. Set `--sklearn 0`
to skip sklearn models, `-qe 0` to skip the quantum model, `-e 0` to skip the
classical model.

### Output / CFA Results

Each run creates a timestamped folder under `data/`:

```
data/run_YYYYMMDD_HHMMSS/
  results.csv              # per-model accuracy, MCC, timing
  cfa_results.csv          # CFA fusion scores (all combinations x methods)
  pred_<MODEL>_val.csv     # per-model validation predictions
  pred_<MODEL>_test.csv    # per-model test predictions
```

**CFA results** are printed to the console at the end of training and saved to
`cfa_results.csv` in the run folder. The CSV contains columns: `combination`,
`method`, `n_models`, `score`. The top-15 results are shown in the terminal.

### Actions

| Flag | Action |
|------|--------|
| `-o train` | Train the model(s) and run CFA |
| `-o cfa` | Run CFA on a previous run (no training) |
| `-o test` | Evaluate on val/test sets |
| `-o pred` | Save predictions to file |
| `-o adv` | Evaluate adversarial robustness |
| `-o latent` | Extract latent representations |

### CFA Options

| Flag | Description | Default |
|------|-------------|---------|
| `--cfa_time` | Time limit in seconds for CFA (0=unlimited) | `1.0` |
| `--cfa_models` | Comma-separated model names to include | all |
| `--run_dir` | Path to a previous run directory (for `-o cfa`) | most recent |

Run CFA standalone on the most recent run:

```bash
uv run --python .venv/bin/python pred_lstm.py -o cfa
```

Run CFA on a specific run with selected models:

```bash
uv run --python .venv/bin/python pred_lstm.py -o cfa --run_dir data/run_20260211_000403 --cfa_models "RANDOM FOREST,GRADIENT BOOST,CLASSICAL ALSTM"
```

CFA evaluates combinations smallest-first and stops when the time limit is
reached. With many models, increase `--cfa_time` to evaluate more combinations.

### Sklearn Model Options

| Flag | Description | Default |
|------|-------------|---------|
| `--sklearn` | Run sklearn models: RF, GB, MLP, LR (0/1) | `0` |

Sklearn models auto-calibrate training set size via a pilot fit to stay within
their time budget (0.8s by default). They flatten the 3D stock data to 2D for
classification.

### Classical ALSTM Options

| Flag | Description | Default |
|------|-------------|---------|
| `-p` | Path to price data | `./data/stocknet-dataset/price/ourpped` |
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
| `-q` | Path to load model from | `./data/saved_model/acl18_alstm/exp` |
| `-qs` | Path to save model to | `./data/tmp/model` |

### Quantum LSTM Options

| Flag | Description | Default |
|------|-------------|---------|
| `-qe` | Quantum LSTM epochs (0 to skip) | `0` |
| `-qi` | Compressed input dimension | `2` |
| `-qh` | Quantum hidden size | `2` |
| `-qd` | VQC circuit depth | `1` |
| `-qt` | Time budget in seconds | `10` |

The quantum model auto-calibrates by timing a pilot batch, then subsamples
training/validation/test data to fit within the time budget. Fewer qubits
(`-qi` + `-qh`) and lower depth (`-qd`) run faster.

### Run a Single Model

Use `-m` / `--model` to run just one model (default: `all`):

| `-m` value | Model |
|------------|-------|
| `all` | Run all enabled models |
| `rf` | Random Forest |
| `gb` | Gradient Boosting |
| `mlp` | MLP Classifier |
| `lr` | Logistic Regression |
| `quantum` | Quantum LSTM |
| `classical` | Classical ALSTM |
| `oom` | OOM Test |

```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m rf
uv run --python .venv/bin/python pred_lstm.py -o train -m quantum -qt 15
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -e 10
```

### Guard Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mem_limit` | Per-model RAM limit in GB (0=off) | `1.0` |
| `--time_limit` | Per-model time limit in seconds (0=off) | `1.0` |
| `--test_oom` | Run OOM test model to verify MemoryGuard (0/1) | `0` |
| `--oom_gb` | GB the OOM test model tries to allocate | `2.0` |
| `-t` | Hard timeout — kill entire script after N seconds (0=off) | `0` |

Models that exceed their RAM or time limit are killed and recorded with zero
accuracy. Killed models are excluded from CFA fusion.

### Examples

Full pipeline with all models (sklearn + quantum + classical + CFA):

```bash
uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1 -qe 3 -e 10 -qt 15
```

Classical only, 10 epochs with adversarial training:

```bash
uv run --python .venv/bin/python pred_lstm.py -o train -e 10 -v 1
```

Sklearn models only with 30s time limit, no classical/quantum:

```bash
uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1 -qe 0 -e 0 --time_limit 30
```

Minimal quantum test (2 qubits, fastest possible):

```bash
uv run --python .venv/bin/python pred_lstm.py -o train -qe 1 -qi 1 -qh 1 -qd 1 -qt 5 -e 0
```

Debug run (1 epoch each, verify all guards and CFA):

```bash
uv run --python .venv/bin/python pred_lstm.py -o train --sklearn 1 -qe 1 -e 1 --time_limit 30
```

## Hyperparameter Recipes

### ACL18 dataset

**LSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -a 0 -l 10 -u 32 -l2 10 -f 1
```

**ALSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -l 5 -u 4 -l2 1 -f 1
```

**Adv-ALSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -l 5 -u 4 -l2 1 -v 1 -rl 1 -q ./data/saved_model/acl18_alstm/exp -la 0.01 -le 0.05
```

### KDD17 dataset

**LSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -p ./data/kdd17/ourpped/ -l 5 -u 4 -l2 0.001 -a 0 -f 1
```

**ALSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -p ./data/kdd17/ourpped/ -l 15 -u 16 -l2 0.001 -f 1
```

**Adv-ALSTM:**
```bash
uv run --python .venv/bin/python pred_lstm.py -p ./data/kdd17/ourpped/ -l 15 -u 16 -l2 0.001 -v 1 -rl 1 -q ./data/saved_model/kdd17_alstm/model -la 0.05 -le 0.001 -f 1
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
