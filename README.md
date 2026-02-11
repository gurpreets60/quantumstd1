# quantumstd1

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or 3.13 (TensorFlow does not yet support 3.14+).

```bash
uv python install 3.13
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python tensorflow tf-keras scikit-learn scipy numpy psutil rich
```

## Run

```bash
uv run --python .venv/bin/python pred_lstm.py -p ./data/stocknet-dataset/price/ourpped -o train
```

### Actions

| Flag | Action |
|------|--------|
| `-o train` | Train the model |
| `-o test` | Evaluate on val/test sets |
| `-o pred` | Save predictions to file |
| `-o adv` | Evaluate adversarial robustness |
| `-o latent` | Extract latent representations |

### Options

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

### Example

Train for 10 epochs with adversarial training:

```bash
uv run --python .venv/bin/python pred_lstm.py -p ./data/stocknet-dataset/price/ourpped -o train -e 10 -v 1
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
