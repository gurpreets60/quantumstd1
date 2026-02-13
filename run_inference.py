#!/usr/bin/env python3
"""Run inference with saved models and CFA on the full dataset.

Uses:
- Sklearn models (PERCEPTRON, PASSIVE AGGRESSIVE, BAGGING DT2) from full-trained run
- Quantum LSTM model from the best available checkpoint
- Runs CFA greedy on the 4-model ensemble
"""
import os
import sys
import shutil
import pickle
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from models.classical_alstm import load_cla_data
from models.quantum_lstm import QLSTMModel

# ---------- Config ----------
FULL_RUN = 'data/run_20260213_044849'       # sklearn models trained on full 20,315
QUANTUM_RUN = 'data/run_20260213_153514'    # latest quantum LSTM checkpoint (5-min saves)
SKLEARN_MODELS = ['SGD MODHUBER', 'PASSIVE AGGRESSIVE', 'QDA',
                  'GRADIENT BOOST', 'EXTRA TREES', 'DECISION TREE', 'BAGGING DT2']
DATA_PATH = './data/stocknet-dataset/price/ourpped'
TRA_DATE = '2014-01-02'
VAL_DATE = '2015-08-03'
TES_DATE = '2015-10-01'
SEQ = 5

# Quantum LSTM architecture (must match saved checkpoint)
QLSTM_INPUT = 3
QLSTM_HIDDEN = 2
QLSTM_DEPTH = 2

# ---------- Setup output dir ----------
out_dir = 'data/run_inference'
os.makedirs(out_dir, exist_ok=True)

# ---------- Load dataset ----------
print('Loading dataset...')
tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt = \
    load_cla_data(DATA_PATH, TRA_DATE, VAL_DATE, TES_DATE, seq=SEQ)

print(f'  Train: {tra_pv.shape[0]}, Val: {val_pv.shape[0]}, Test: {tes_pv.shape[0]}')
print(f'  Features: {tra_pv.shape[2]}')

# ---------- Sklearn inference ----------
print('\n=== Sklearn Inference ===')
for model_name in SKLEARN_MODELS:
    safe = model_name.replace(' ', '_')
    pkl_path = os.path.join(FULL_RUN, f'model_{safe}.pkl')
    print(f'Loading {model_name} from {pkl_path}...')
    with open(pkl_path, 'rb') as f:
        estimator = pickle.load(f)

    # Sklearn models expect 2D: (n_samples, n_features)
    # Flatten the (n_samples, seq, features) -> (n_samples, seq*features)
    val_flat = val_pv.reshape(val_pv.shape[0], -1)
    tes_flat = tes_pv.reshape(tes_pv.shape[0], -1)

    # Use decision_function if available (hinge mode), else predict
    if hasattr(estimator, 'decision_function'):
        val_pred = estimator.decision_function(val_flat)
        tes_pred = estimator.decision_function(tes_flat)
    else:
        val_pred = estimator.predict(val_flat)
        tes_pred = estimator.predict(tes_flat)

    # Save predictions in same format as training pipeline
    val_gt_arr = val_gt
    tes_gt_arr = tes_gt
    header = 'epoch,prediction,ground_truth'

    val_out = np.column_stack([
        np.zeros(len(val_pred)),  # epoch 0
        val_pred,
        val_gt_arr
    ])
    tes_out = np.column_stack([
        np.zeros(len(tes_pred)),
        tes_pred,
        tes_gt_arr
    ])

    val_csv = os.path.join(out_dir, f'pred_{safe}_val.csv')
    tes_csv = os.path.join(out_dir, f'pred_{safe}_test.csv')
    np.savetxt(val_csv, val_out, delimiter=',', header=header, comments='')
    np.savetxt(tes_csv, tes_out, delimiter=',', header=header, comments='')

    # Quick accuracy check
    val_binary = (val_pred > 0).astype(int)
    tes_binary = (tes_pred > 0).astype(int)
    val_acc = np.mean(val_binary == val_gt_arr)
    tes_acc = np.mean(tes_binary == tes_gt_arr)
    print(f'  {model_name}: val_acc={val_acc:.4f}, test_acc={tes_acc:.4f}')

# ---------- Quantum LSTM inference ----------
print('\n=== Quantum LSTM Inference ===')
input_size = tra_pv.shape[2]
print(f'Building QLSTMModel: input={input_size}, compress->{QLSTM_INPUT}, '
      f'hidden={QLSTM_HIDDEN}, depth={QLSTM_DEPTH}')

model = QLSTMModel(input_size, QLSTM_HIDDEN, 1, QLSTM_DEPTH,
                   qlstm_input=QLSTM_INPUT).double()

# Load saved state dict
qlstm_pkl = os.path.join(QUANTUM_RUN, 'model_QUANTUM_LSTM.pkl')
print(f'Loading quantum weights from {qlstm_pkl}...')
with open(qlstm_pkl, 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model.eval()

# Convert data to tensors
val_x = torch.from_numpy(val_pv).double()
tes_x = torch.from_numpy(tes_pv).double()
val_gt_arr = val_gt
tes_gt_arr = tes_gt

# Run inference in batches to manage memory
batch_size = 64
print(f'Running quantum inference on {val_x.size(0)} val + {tes_x.size(0)} test samples...')

def batch_predict(model, x, batch_size=64):
    preds = []
    n = x.size(0)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        with torch.no_grad():
            out, _ = model(x[start:end])
            preds.append(out[:, -1, :].numpy())
        if (start // batch_size) % 50 == 0:
            print(f'  ... {start}/{n} samples', flush=True)
    return np.vstack(preds)

print('Val set:')
val_pred = batch_predict(model, val_x, batch_size).flatten()
print('Test set:')
tes_pred = batch_predict(model, tes_x, batch_size).flatten()

# Save predictions
val_out = np.column_stack([np.zeros(len(val_pred)), val_pred, val_gt_arr])
tes_out = np.column_stack([np.zeros(len(tes_pred)), tes_pred, tes_gt_arr])
np.savetxt(os.path.join(out_dir, 'pred_QUANTUM_LSTM_val.csv'),
           val_out, delimiter=',', header='epoch,prediction,ground_truth', comments='')
np.savetxt(os.path.join(out_dir, 'pred_QUANTUM_LSTM_test.csv'),
           tes_out, delimiter=',', header='epoch,prediction,ground_truth', comments='')

val_binary = (val_pred > 0.5).astype(int)
tes_binary = (tes_pred > 0.5).astype(int)
val_acc = np.mean(val_binary == val_gt_arr)
tes_acc = np.mean(tes_binary == tes_gt_arr)
print(f'  QUANTUM LSTM: val_acc={val_acc:.4f}, test_acc={tes_acc:.4f}')

# ---------- Run CFA ----------
print('\n=== Running CFA Greedy ===')
from pred_lstm import run_cfa_greedy
run_cfa_greedy(out_dir, summary=None, time_limit=60.0,
               models_filter=['QUANTUM LSTM', 'PERCEPTRON', 'PASSIVE AGGRESSIVE', 'BAGGING DT2'])

# Print final results
greedy_csv = os.path.join(out_dir, 'cfa_greedy.csv')
if os.path.exists(greedy_csv):
    print('\n=== CFA GREEDY RESULTS ===')
    with open(greedy_csv) as f:
        print(f.read())
