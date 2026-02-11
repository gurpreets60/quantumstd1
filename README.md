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
| `-o test` | Evaluate on val/test sets |
| `-o pred` | Save predictions to file |
| `-o adv` | Evaluate adversarial robustness |
| `-o latent` | Extract latent representations |

### Sklearn Model Options

| Flag | Description | Default |
|------|-------------|---------|
| `--sklearn` | Run sklearn models: Random Forest, Gradient Boosting, MLP (0/1) | `0` |

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
