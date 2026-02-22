# quantumstd1

Quantum and classical models for stock movement prediction. Auto-discovers all
sklearn models in `models/` (128+), plus Quantum LSTM, two batch quantum models
(Batch QLSTM and Batch VQFWP), and Classical ALSTM. Fuses predictions with
Combinatorial Fusion Analysis (CFA) using greedy forward selection to find the
best ensemble. Each model is guarded by per-model RAM and time limits and
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
# all sklearn models (auto-discovered, fast with default 1s time limit)
uv run --python .venv/bin/python pred_lstm.py -o train -m sklearn

# full pipeline: all sklearn + quantum (1 epoch) + classical (1 epoch)
uv run --python .venv/bin/python pred_lstm.py -o train -m all -qe 1 -e 1 --time_limit 30

# only the two batch quantum models
uv run --python .venv/bin/python pred_lstm.py -o train -m quantum_batch --qbatch_epoch 1 --qbatch_time 8 --time_limit 90
./run_quantum_batch.sh 1 8 90

# run CFA on previous results (instant, no training)
uv run --python .venv/bin/python pred_lstm.py -o cfa
```

### Output

Each run creates a timestamped folder under `data/`:

```
data/run_YYYYMMDD_HHMMSS/
  results.csv              # per-model accuracy, MCC, timing
  cfa_greedy.csv           # greedy CFA: round-by-round ensemble building (default)
  cfa_results.csv          # exhaustive CFA: all combinations x methods
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

Use `-m` / `--model` to select which models to train:

| `-m` value | What runs |
|------------|-----------|
| `all` | All sklearn models + quantum + classical (default) |
| `sklearn` | All auto-discovered sklearn models only |
| `quantum` | Quantum LSTM only |
| `quantum_batch` | Batch QLSTM + Batch VQFWP only |
| `classical` | Classical ALSTM only |
| `oom` | OOM Test (for verifying MemoryGuard) |

All sklearn models in the `models/` directory are auto-discovered at import
time. Any `.py` file that defines a `SklearnTrainer` subclass is automatically
included -- no manual registration needed.

```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m sklearn
uv run --python .venv/bin/python pred_lstm.py -o train -m quantum -qt 15 --time_limit 30
uv run --python .venv/bin/python pred_lstm.py -o train -m quantum_batch --qbatch_epoch 1 --qbatch_time 8 --time_limit 90
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -e 10 --time_limit 0
```

## CFA (Combinatorial Fusion Analysis)

CFA fuses predictions from multiple models. Runs automatically after training,
or standalone with `-o cfa`. Two modes are available:

- **Greedy** (default): Forward selection with pruning. Builds the best ensemble
  incrementally. Saves progress after each round to `cfa_greedy.csv` so partial
  results survive crashes. Scales to 100+ models.
- **Exhaustive**: Tests all 2^N combinations smallest-first. Fast for <10 models
  but explodes combinatorially. Saves to `cfa_results.csv`.

```bash
# CFA on most recent run (greedy, default)
uv run --python .venv/bin/python pred_lstm.py -o cfa

# CFA exhaustive mode (for small model counts)
uv run --python .venv/bin/python pred_lstm.py -o cfa --cfa_mode exhaustive

# CFA on a specific run
uv run --python .venv/bin/python pred_lstm.py -o cfa --run_dir data/run_20260211_000403

# CFA with selected models only
uv run --python .venv/bin/python pred_lstm.py -o cfa --cfa_models "RANDOM FOREST,GRADIENT BOOST"

# CFA with longer time limit (for many models)
uv run --python .venv/bin/python pred_lstm.py -o cfa --cfa_time 5
```

| Flag | Description | Default |
|------|-------------|---------|
| `--cfa_mode` | CFA algorithm: `greedy` or `exhaustive` | `greedy` |
| `--cfa_time` | Time limit for CFA in seconds (0=unlimited) | `1.0` |
| `--cfa_models` | Comma-separated model names to include | all |
| `--run_dir` | Path to a previous run directory | most recent |

### How CFA Scores Are Calculated

**1. Prediction collection.** Each model saves raw continuous predictions (not
binary) for val and test sets to CSV files. Sklearn models output
`predict_proba` or `decision_function` scores; deep learning models output
final-layer activations.

**2. Alignment.** CFA requires every model to have predictions on the same
samples. The loader truncates all models to the length of the shortest
prediction file. If Quantum LSTM only predicted 56 samples, all models are
truncated to 56.

**3. Normalization.** Predictions are min-max normalized using **validation-set
statistics** (min/max from val predictions). This prevents test data from
leaking into normalization.

**4. Fusion.** For each subset of 2+ models, six methods produce a fused score:

| Method | Description |
|--------|-------------|
| ASC    | Average Score Combination -- mean of normalized scores |
| ARC    | Average Rank Combination -- mean of per-model ranks |
| WSCDS  | Weighted Score by Cognitive Diversity Strength |
| WRCDS  | Weighted Rank by Cognitive Diversity Strength |
| WSCP   | Weighted Score by individual model performance (on val) |
| WRCP   | Weighted Rank by individual model performance (on val) |

Diversity weights come from Rank-Score Characteristic (RSC) curves:
CD = sqrt(sum((RSC_A - RSC_B)^2) / n). Diversity Strength (DS) is the mean CD
of a model against all others.

**5. Evaluation.** The fused score is thresholded at 0.5 (or median for rank
methods) to produce binary predictions, scored with `accuracy_score`.

### How Greedy Selection Discovers Combos

The greedy algorithm builds the best ensemble one model at a time:

```
Pool: [all N trained models, sorted by individual val accuracy]
Best combo: [top individual model]

Repeat until no improvement or time runs out:
  1. FORWARD:  Try adding each remaining model to current best combo.
               Pick the addition that improves val score the most.
  2. EVALUATE: Score the new combo on val set across all 6 CFA methods.
               Take the best method's score as the combo's score.
  3. PRUNE:    Among models that never helped any combo this round,
               drop the one with worst individual performance.
```

**Why this finds good combos.** CFA's value comes from combining diverse models.
A model that is individually mediocre (e.g. Nearest Centroid at 50.5% accuracy)
can still improve an ensemble if its errors are uncorrelated with the other
members. The greedy search tests every candidate against the current combo each
round, so it catches these diversity contributions. The pruning step only
removes models that **never helped in any combo** -- it targets redundancy, not
weakness.

**Complexity.** Each round tests at most N candidates (one CFA eval each).
With K rounds and pruning, total evaluations are O(N*K) -- linear, not
exponential. For 50 models and 8 rounds this is ~400 evaluations, done in
under 2 seconds. Exhaustive search of 2^50 combinations would take longer
than the age of the universe.

**Incremental saves.** After each round, the current state is written to
`cfa_greedy.csv`. If the process is killed, you keep all completed rounds and
can re-run CFA on the same run directory to continue.

## Discovering the Best Ensemble: Walkthrough

This is the end-to-end workflow for finding the best model combination.

### Step 1: Train all models

```bash
# Smoke test (fast, strict limits -- confirms everything runs)
uv run --python .venv/bin/python pred_lstm.py -o train -m sklearn --time_limit 0.5 -t 90

# Production run (generous limits -- models train on full data)
uv run --python .venv/bin/python pred_lstm.py -o train -m sklearn --time_limit 10 -t 0
```

With `--time_limit 0.5`, many models get killed by TimeGuard. That's fine for
a smoke test -- the surviving models still get fused. For a real evaluation,
use `--time_limit 10` or higher so models train on the full 20K-sample dataset.

### Step 2: Check individual results

Open `results.csv`. Sort by `val_acc`:

```
model                 val_acc  test_acc  status
RIDGE ALPHA10         0.5746   0.4884    DONE
LINEARSVC L2          0.5734   0.4898    DONE
EXTRA TREE SINGLE     0.5652   0.5175    DONE
RANDOM FOREST         0.5605   0.5156    DONE
...
GAUSSIAN PROCESS      0.0000   0.0000    DONE     <- killed by MemoryGuard
```

Models with 0.0000 were killed by a guard. They have no prediction files and
are automatically excluded from CFA.

**What to look for:**
- Models with high `val_acc` but low `test_acc` (e.g. RIDGE at 0.57/0.49) are
  overfitting. They may still help in an ensemble if their errors are diverse.
- Models with moderate `val_acc` but matching `test_acc` (e.g. GAUSSIAN NB at
  0.53/0.52) are more reliable individually.
- Models clustered at ~50% are near random -- but even these can add ensemble
  diversity if their predictions are uncorrelated with others.

### Step 3: Read the greedy CFA log

Open `cfa_greedy.csv`. The file has three sections:

**Individual scores (round 0, action=individual).** Every usable model ranked
by val accuracy. This is the starting pool.

**Selection rounds (action=add).** Each row adds one model to the ensemble:

```
round  action  combination                           method  val_score  test_score  pool_size
0      start   EXTRA TREES                           ASC     0.5836     0.5081      59
1      add     EXTRA TREES+BAGGING                   ASC     0.5894     0.5151      57
2      add     EXTRA TREES+BAGGING+EXTRA TREES DEEP  WSCDS   0.6078     0.5016      55
3      add     ...+CALIBRATED SGD                    ASC     0.6137     0.5032      53
4      add     ...+CALIBRATED RIDGE                  ASC     0.6239     0.5097      51
```

**Stop row (action=stop_no_gain).** The algorithm tried all remaining candidates
and none improved the score. The ensemble is complete.

**What to look for:**

1. **The val_score column should increase each round.** That's the greedy
   guarantee -- each addition must improve the best CFA method's score.

2. **Watch the val-test gap.** If val_score keeps climbing but test_score
   stays flat or drops, the ensemble is overfitting to the 2,555-sample
   validation set. The last round where test_score also improved is likely the
   sweet spot.

3. **The method column tells you what fusion works.** If it switches from ASC
   to WSCDS at round 3, that means the diversity-weighted fusion became better
   than simple averaging once enough diverse models joined. WSCDS and WRCDS
   tend to win when models have genuinely different error patterns.

4. **Pruned models are diagnostic.** If a model is pruned, it never improved
   any combo it joined -- it's redundant with what's already in the pool. If
   many models of the same family are pruned (e.g. all KNN variants), that
   family doesn't add unique signal.

5. **Pool size shrinks from both ends.** Each round removes one model (added
   to combo) and may prune one more. When the pool is small, pruning stops
   to avoid discarding useful candidates.

### Step 4: Explore alternatives with targeted CFA

The greedy result is one good ensemble, but it might not be optimal. Use
`--cfa_models` to test specific hypotheses:

```bash
# Test a specific combo you think might work
uv run --python .venv/bin/python pred_lstm.py -o cfa \
  --cfa_models "RANDOM FOREST,GAUSSIAN NB,BERNOULLI RBM PIPE" \
  --cfa_mode exhaustive

# Compare tree-based vs linear ensembles
uv run --python .venv/bin/python pred_lstm.py -o cfa \
  --cfa_models "RANDOM FOREST,EXTRA TREES,GRADIENT BOOST,BAGGING DT"

uv run --python .venv/bin/python pred_lstm.py -o cfa \
  --cfa_models "RIDGE CLASSIFIER,LOGISTIC ELASTIC,PASSIVE AGGRESSIVE,LDA"

# Take the greedy result and try swapping one member
uv run --python .venv/bin/python pred_lstm.py -o cfa \
  --cfa_models "EXTRA TREES,BAGGING,CALIBRATED SGD,GAUSSIAN NB"
```

**Strategy: diverse families beat similar models.** CFA rewards error
diversity. An ensemble of Random Forest + Naive Bayes + Ridge (three different
learning paradigms) usually beats Random Forest + Extra Trees + Bagging
(three tree ensembles with correlated errors). Look at the pruned models in
`cfa_greedy.csv` -- if all tree models were pruned after one tree entered the
combo, that confirms tree models are redundant with each other.

### Step 5: Validate the final pick

Compare `val_score` vs `test_score` for your chosen ensemble:

| Gap | Interpretation |
|-----|---------------|
| < 2% | Good -- ensemble generalizes well |
| 2-5% | Acceptable for 2,555-sample val set |
| > 5% | Likely overfitting -- consider a simpler ensemble |
| test > val | Unusual but possible with small samples; don't trust it |

For publication-grade results, the single fixed train/val/test split is not
sufficient. See "Data Split" below for caveats.

## Data Split

The split is **temporal and non-leaking**:

- **Train:** 2014-01-02 to 2015-08-03 (20,315 instances)
- **Val:** 2015-08-03 to 2015-10-01 (2,555 instances)
- **Test:** 2015-10-01 to end of dataset (3,720 instances)

All 87 tickers share the same date boundaries. No future data leaks into
training. However, this is a **single fixed split**, not a rolling walk-forward
backtest -- models do not retrain on expanding windows.

## Known Caveats

### 1. Training subsampling hides model quality

Auto-calibration caps training samples to fit each model's time budget. With
`--time_limit 1`, Random Forest trains on ~907/20,315 samples. Accuracy on 5%
of data is not representative. Use `--time_limit 10` or higher for real evals.

### 2. Quantum LSTM truncates CFA eval set

Quantum LSTM caps its val/test set to fit its time budget (e.g. 56 out of 3,720
test samples). Because alignment truncates to the shortest model, the entire
CFA evaluation drops to 56 samples. At n=56, one extra correct sample changes
accuracy by ~1.8%. Run Quantum LSTM with generous time (`-qt 60`) or exclude it
from CFA with `--cfa_models`.

### 3. Narrow validation window

The val set covers only 2 months (2,555 samples across 87 tickers). CFA's
WSCP/WRCP methods use val accuracy as fusion weights. With many models and
many greedy rounds, the risk of overfitting to this window increases. Watch
the val-test gap.

### 4. Exhaustive CFA explodes combinatorially

| Models | Combinations | 1s budget covers |
|--------|-------------|------------------|
| 6      | 63          | All              |
| 17     | 131,071     | ~348             |
| 50     | ~1 quadrillion | ~350          |

The greedy mode avoids this entirely. Use exhaustive only for <10 models.

## Guard Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mem_limit` | Per-model RAM limit in GB (0=off) | `1.0` |
| `--time_limit` | Per-model time limit in seconds (0=off) | `1.0` |
| `--test_oom` | Run OOM test model to verify MemoryGuard (0/1) | `0` |
| `--oom_gb` | GB the OOM test tries to allocate | `2.0` |
| `-t` | Hard timeout -- kill entire script after N seconds (0=off) | `0` |

Models that exceed their limit are killed and excluded from CFA.

**Note:** The default `--time_limit 1` is strict. Use `--time_limit 30` for
quantum/classical models that need more time, or `--time_limit 0` to disable.

## Model-Specific Options

### Sklearn Models

All sklearn models in `models/` are auto-discovered at startup. No flags needed
-- use `-m sklearn` or `-m all` to include them. Auto-calibrates training set
size via pilot fit to stay within each model's time budget.

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

### Batch Quantum Models (QLSTM + VQFWP)

| Flag | Description | Default |
|------|-------------|---------|
| `--qbatch_epoch` | Epochs for both batch quantum models (0 to skip) | `0` |
| `--qbatch_time` | Time budget in seconds for each batch quantum model | `8` |

Use `-m quantum_batch` to run only these two models:

```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m quantum_batch --qbatch_epoch 1 --qbatch_time 8 --time_limit 90
./run_quantum_batch.sh 1 8 90
```

## Examples

```bash
# All sklearn models (auto-discovered, fast)
uv run --python .venv/bin/python pred_lstm.py -o train -m sklearn

# Full pipeline: all sklearn + quantum (3 epochs) + classical (10 epochs)
uv run --python .venv/bin/python pred_lstm.py -o train -m all -qe 3 -e 10 -qt 15 --time_limit 30

# Classical only, 10 epochs with adversarial training
uv run --python .venv/bin/python pred_lstm.py -o train -m classical -e 10 -v 1 --time_limit 0

# Minimal quantum test (2 qubits, fastest possible)
uv run --python .venv/bin/python pred_lstm.py -o train -m quantum -qe 1 -qi 1 -qh 1 -qd 1 -qt 5

# Run only the two batch quantum models
uv run --python .venv/bin/python pred_lstm.py -o train -m quantum_batch --qbatch_epoch 1 --qbatch_time 8 --time_limit 90

# Debug run (1 epoch each, all models)
uv run --python .venv/bin/python pred_lstm.py -o train -m all -qe 1 -e 1 --time_limit 30

# CFA on previous results (greedy, default)
uv run --python .venv/bin/python pred_lstm.py -o cfa

# CFA exhaustive on specific run
uv run --python .venv/bin/python pred_lstm.py -o cfa --cfa_mode exhaustive --run_dir data/run_20260211_000403
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
it gets killed. The `SklearnTrainer` base class handles most of the work -- it
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

That's it. The auto-discovery system in `models/__init__.py` will find any
`SklearnTrainer` subclass automatically -- no imports or wiring needed.

**2. Test it**

```bash
uv run --python .venv/bin/python pred_lstm.py -o train -m sklearn --time_limit 1 --mem_limit 1
```

### Fitting within 1s / 1GB limits

The `SklearnTrainer` base class automatically subsamples training data to fit
the time budget. But if your model is still too slow or uses too much RAM:

| Problem | Fix |
|---------|-----|
| Killed by TimeGuard | Reduce complexity: fewer estimators (`n_estimators=20`), shallower trees (`max_depth=3`), fewer iterations (`max_iter=50`) |
| Killed by MemoryGuard | Use lighter model, reduce `n_estimators`, set `max_depth`, avoid algorithms that build large internal structures (e.g. KNN with large k) |
| Pilot fit alone takes >0.5s | The base class only uses 50 samples for pilot -- if that's too slow, simplify the estimator or use a faster solver |
| MLP convergence warning | Lower `max_iter` or use a simpler architecture (`hidden_layer_sizes=(16,)`) |

**Rule of thumb**: start with minimal parameters, test with `-m sklearn --time_limit 1`,
and increase complexity only while it still passes. The auto-calibration will
subsample training data as needed, so low `n_estimators` / `max_depth` /
`max_iter` is the main lever.

### Example: Logistic Regression (simplest possible model)

`models/logistic_regression.py` -- 8 lines total:

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

Runs in 0.2s, 0 MB above baseline -- well within limits.
