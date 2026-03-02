#!/usr/bin/env python3
"""Generate CFA analysis plots for the LaTeX report."""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt

sys.path.insert(0, os.path.dirname(__file__))

PLOT_DIR = 'report_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 9

# ── Cognitive Diversity helper ──
def cognitive_diversity(norm_a, ranks_a, norm_b, ranks_b):
    n = len(norm_a)
    scores_a = {r: s for r, s in zip(ranks_a, norm_a)}
    scores_b = {r: s for r, s in zip(ranks_b, norm_b)}
    div_sum = sum((scores_a.get(r, 0) - scores_b.get(r, 0))**2 for r in scores_a)
    return sqrt(div_sum / n) if n > 0 else 0

# ── Load full-dataset sklearn predictions (26 models) ──
FULL_RUN_DIR = 'data/run_20260213_044849'
models_data = {}
for vp in sorted(os.listdir(FULL_RUN_DIR)):
    if vp.startswith('pred_') and vp.endswith('_val.csv'):
        name = vp.replace('pred_', '').replace('_val.csv', '').replace('_', ' ')
        tp = vp.replace('_val.csv', '_test.csv')
        vd = np.loadtxt(os.path.join(FULL_RUN_DIR, vp), delimiter=',', skiprows=1)
        td = np.loadtxt(os.path.join(FULL_RUN_DIR, tp), delimiter=',', skiprows=1)
        last_e = int(vd[:, 0].max())
        vd = vd[vd[:, 0] == last_e]
        td = td[td[:, 0] == last_e]
        models_data[name] = {
            'val_pred': vd[:, 1], 'val_gt': vd[:, 2],
            'tes_pred': td[:, 1], 'tes_gt': td[:, 2]
        }

print(f'Loaded {len(models_data)} sklearn models from {FULL_RUN_DIR}')

# ── Load inference predictions (7 winning sklearn + quantum) ──
INF_DIR = 'data/run_inference'
inf_data = {}
for vp in sorted(os.listdir(INF_DIR)):
    if vp.startswith('pred_') and vp.endswith('_val.csv'):
        name = vp.replace('pred_', '').replace('_val.csv', '').replace('_', ' ')
        tp = vp.replace('_val.csv', '_test.csv')
        tp_path = os.path.join(INF_DIR, tp)
        if not os.path.exists(tp_path):
            continue
        vd = np.loadtxt(os.path.join(INF_DIR, vp), delimiter=',', skiprows=1)
        td = np.loadtxt(tp_path, delimiter=',', skiprows=1)
        if vd.ndim == 1:
            vd = vd.reshape(1, -1)
        if td.ndim == 1:
            td = td.reshape(1, -1)
        last_e = int(vd[:, 0].max())
        vd = vd[vd[:, 0] == last_e]
        td = td[td[:, 0] == last_e]
        inf_data[name] = {
            'val_pred': vd[:, 1], 'val_gt': vd[:, 2],
            'tes_pred': td[:, 1], 'tes_gt': td[:, 2]
        }

print(f'Loaded {len(inf_data)} models from {INF_DIR}')

# Ground truth from full dataset
model_names = list(models_data.keys())
val_gt = (list(models_data.values())[0]['val_gt'] > 0.5).astype(int)
tes_gt = (list(models_data.values())[0]['tes_gt'] > 0.5).astype(int)

# ── Individual model accuracy (all 26 sklearn) ──
val_accs = {}
tes_accs = {}
for name, d in models_data.items():
    vp = d['val_pred']
    tp = d['tes_pred']
    if np.abs(vp).max() > 2:
        val_accs[name] = np.mean((vp > 0).astype(int) == val_gt)
        tes_accs[name] = np.mean((tp > 0).astype(int) == tes_gt)
    else:
        val_accs[name] = np.mean((vp > 0.5).astype(int) == val_gt)
        tes_accs[name] = np.mean((tp > 0.5).astype(int) == tes_gt)

# ── PLOT 1: Individual Model Performance (bar chart) ──
fig, ax = plt.subplots(figsize=(14, 6))
sorted_models = sorted(model_names, key=lambda m: tes_accs[m], reverse=True)
x = np.arange(len(sorted_models))
w = 0.35
ax.bar(x - w/2, [val_accs[m] for m in sorted_models], w, label='Validation', color='steelblue', alpha=0.8)
ax.bar(x + w/2, [tes_accs[m] for m in sorted_models], w, label='Test', color='coral', alpha=0.8)
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Random baseline (0.5)')
ax.set_xticks(x)
ax.set_xticklabels([m[:12] for m in sorted_models], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Accuracy')
ax.set_title('Individual Model Performance (Full Dataset: 2,555 Val / 3,720 Test)')
ax.legend()
ax.set_ylim(0.44, 0.60)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'individual_performance.pdf'), bbox_inches='tight')
plt.close()
print('Saved: individual_performance.pdf')

# ── RSC and Diversity for the 8-model ensemble (7 sklearn + QUANTUM LSTM) ──
# Truncate all to shortest (quantum) sample count for fair comparison
ensemble_models = ['SGD MODHUBER', 'PASSIVE AGGRESSIVE', 'QDA',
                   'GRADIENT BOOST', 'EXTRA TREES', 'DECISION TREE',
                   'BAGGING DT2', 'QUANTUM LSTM']
ensemble_present = [m for m in ensemble_models if m in inf_data]
print(f'Ensemble models found: {ensemble_present}')

# Find min sample count
min_val = min(len(inf_data[m]['val_pred']) for m in ensemble_present)
min_tes = min(len(inf_data[m]['tes_pred']) for m in ensemble_present)
print(f'Truncating to {min_val} val / {min_tes} test samples')

# Normalize and rank (on truncated test set)
ens_norm_tes = {}
ens_ranks_tes = {}
ens_tes_gt = (inf_data[ensemble_present[0]]['tes_gt'][:min_tes] > 0.5).astype(int)

for name in ensemble_present:
    pred = inf_data[name]['tes_pred'][:min_tes]
    lo, hi = inf_data[name]['val_pred'][:min_val].min(), inf_data[name]['val_pred'][:min_val].max()
    if hi - lo < 1e-10:
        ens_norm_tes[name] = np.zeros(min_tes)
    else:
        ens_norm_tes[name] = (pred - lo) / (hi - lo)
    order = ens_norm_tes[name].argsort()
    r = np.empty_like(order)
    r[order] = np.arange(len(order), 0, -1)
    ens_ranks_tes[name] = r

# ── PLOT 2: RSC for 8-model ensemble (CFA-style) ──
fig, ax = plt.subplots(figsize=(10, 6))
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
colors = plt.cm.Set1(np.linspace(0, 0.95, len(ensemble_present)))
interval = max(1, min_tes // 60)

for i, name in enumerate(ensemble_present):
    s = ens_norm_tes[name]
    rsc = np.sort(s)[::-1]
    rank_positions = np.arange(1, len(rsc) + 1)
    sel = np.arange(0, len(rsc), interval)
    lw = 2.5 if name == 'QUANTUM LSTM' else 1.5
    ms = 7 if name == 'QUANTUM LSTM' else 5
    ax.plot(rank_positions[sel], rsc[sel], marker=markers[i], markersize=ms,
            alpha=0.85, color=colors[i], linewidth=lw, label=name[:15])

ax.set_xlabel('Rank Position')
ax.set_ylabel('Normalized Score (descending)')
ax.set_title(f'Rank-Score Characteristic (RSC) — 8-Model Ensemble ({min_tes} test samples)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rsc_ensemble.pdf'), bbox_inches='tight')
plt.close()
print('Saved: rsc_ensemble.pdf')

# ── PLOT 3: Cognitive Diversity Heatmap for 8-model ensemble ──
n_ens = len(ensemble_present)
cd_matrix = np.zeros((n_ens, n_ens))
for i, m1 in enumerate(ensemble_present):
    for j, m2 in enumerate(ensemble_present):
        if i != j:
            cd_matrix[i, j] = cognitive_diversity(
                ens_norm_tes[m1], ens_ranks_tes[m1],
                ens_norm_tes[m2], ens_ranks_tes[m2])

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cd_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(n_ens))
ax.set_yticks(range(n_ens))
short_labels = [m[:12] for m in ensemble_present]
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(short_labels, fontsize=8)
for i in range(n_ens):
    for j in range(n_ens):
        if i != j:
            ax.text(j, i, f'{cd_matrix[i,j]:.3f}', ha='center', va='center', fontsize=7)
ax.set_title(f'Cognitive Diversity Matrix — 8-Model Ensemble ({min_tes} test samples)')
plt.colorbar(im, ax=ax, label='Cognitive Diversity')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'diversity_heatmap.pdf'), bbox_inches='tight')
plt.close()
print('Saved: diversity_heatmap.pdf')

# ── PLOT 4: CFA Ensemble Performance ──
from pred_lstm import _load_predictions, _cfa_eval_combo
from models import cfa as unified_cfa

data = _load_predictions(FULL_RUN_DIR, summary=None)
val_preds_cfa, tes_preds_cfa, val_gt_cfa, tes_gt_cfa, usable = data

val_df = pd.DataFrame({name: val_preds_cfa[name] for name in usable})
tes_df = pd.DataFrame({name: tes_preds_cfa[name] for name in usable})
norm_val_df = pd.DataFrame(index=val_df.index)
norm_tes_df = pd.DataFrame(index=tes_df.index)
for c in usable:
    lo, hi = val_df[c].min(), val_df[c].max()
    norm_val_df[c] = unified_cfa.normalize_minmax(val_df[c].values, lo, hi)
    norm_tes_df[c] = unified_cfa.normalize_minmax(tes_df[c].values, lo, hi)

_, ds, _ = unified_cfa.compute_cd_matrix(val_df)
perf = {c: unified_cfa.evaluate(norm_val_df[c].values, val_gt_cfa, 'accuracy') for c in usable}

methods = unified_cfa.ALL_METHODS
key_combos = [
    ['SGD MODHUBER'],
    ['QDA'],
    ['GRADIENT BOOST'],
    ['BAGGING DT2'],
    ['GRADIENT BOOST', 'EXTRA TREES'],
    ['GRADIENT BOOST', 'QDA'],
    ['SGD MODHUBER', 'PASSIVE AGGRESSIVE', 'BAGGING DT2'],
    ['GRADIENT BOOST', 'QDA', 'SGD MODHUBER'],
]

combo_results = []
for combo in key_combos:
    if all(c in usable for c in combo):
        for method in methods:
            val_score = _cfa_eval_combo(combo, norm_val_df, val_gt_cfa, ds, perf)
            tes_score = _cfa_eval_combo(combo, norm_tes_df, tes_gt_cfa, ds, perf)
            combo_results.append({
                'combo': '+'.join([c[:8] for c in combo]),
                'n': len(combo),
                'val': val_score[0],
                'test': tes_score[0],
                'method': val_score[1]
            })
            break

fig, ax = plt.subplots(figsize=(12, 6))
combo_labels = [r['combo'] for r in combo_results]
val_scores = [r['val'] for r in combo_results]
tes_scores = [r['test'] for r in combo_results]
x = np.arange(len(combo_labels))
w = 0.35
ax.bar(x - w/2, val_scores, w, label='Validation', color='steelblue', alpha=0.8)
ax.bar(x + w/2, tes_scores, w, label='Test', color='coral', alpha=0.8)
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Random baseline')
ax.set_xticks(x)
ax.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Accuracy')
ax.set_title('CFA Ensemble Performance (Best Method per Combination)')
ax.legend()
ax.set_ylim(0.48, 0.60)
ax.grid(axis='y', alpha=0.3)
for i, (v, t) in enumerate(zip(val_scores, tes_scores)):
    ax.text(i - w/2, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    ax.text(i + w/2, t + 0.002, f'{t:.3f}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'cfa_ensemble_performance.pdf'), bbox_inches='tight')
plt.close()
print('Saved: cfa_ensemble_performance.pdf')

# ── PLOT 5: Quantum LSTM Training Convergence ──
epochs = [0, 1, 2, 3]
losses = [0.2515, 0.2494, 0.2490, 0.2491]
val_acc_q = [0.4763, 0.4751, 0.4599, 0.4669]
tes_acc_q = [0.5054, 0.5121, 0.5164, 0.5151]
epoch_time_h = [2.68, 2.48, 2.29, 2.27]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

axes[0].plot(epochs, losses, 'o-', color='darkblue', linewidth=2, markersize=8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss (MSE)')
axes[0].set_title('Quantum LSTM Training Loss')
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0.248, 0.253)

axes[1].plot(epochs, val_acc_q, 's-', color='steelblue', linewidth=2, markersize=8, label='Validation')
axes[1].plot(epochs, tes_acc_q, 'D-', color='coral', linewidth=2, markersize=8, label='Test')
axes[1].axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Quantum LSTM Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0.44, 0.54)

axes[2].bar(epochs, epoch_time_h, color='mediumpurple', alpha=0.8)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Time (hours)')
axes[2].set_title('Epoch Duration')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Quantum LSTM (5 qubits, depth=2) — Full Dataset Training (20,315 samples)', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'quantum_training.pdf'), bbox_inches='tight')
plt.close()
print('Saved: quantum_training.pdf')

print('\nAll plots generated in:', PLOT_DIR)
