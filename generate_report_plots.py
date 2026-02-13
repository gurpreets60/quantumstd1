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

# ── Load full-dataset sklearn predictions ──
RUN_DIR = 'data/run_20260213_044849'
models_data = {}
for vp in sorted(os.listdir(RUN_DIR)):
    if vp.startswith('pred_') and vp.endswith('_val.csv'):
        name = vp.replace('pred_', '').replace('_val.csv', '').replace('_', ' ')
        tp = vp.replace('_val.csv', '_test.csv')
        vd = np.loadtxt(os.path.join(RUN_DIR, vp), delimiter=',', skiprows=1)
        td = np.loadtxt(os.path.join(RUN_DIR, tp), delimiter=',', skiprows=1)
        last_e = int(vd[:, 0].max())
        vd = vd[vd[:, 0] == last_e]
        td = td[td[:, 0] == last_e]
        models_data[name] = {
            'val_pred': vd[:, 1], 'val_gt': vd[:, 2],
            'tes_pred': td[:, 1], 'tes_gt': td[:, 2]
        }

print(f'Loaded {len(models_data)} models')
model_names = list(models_data.keys())

# Ground truth
val_gt = (list(models_data.values())[0]['val_gt'] > 0.5).astype(int)
tes_gt = (list(models_data.values())[0]['tes_gt'] > 0.5).astype(int)

# ── Individual model accuracy ──
val_accs = {}
tes_accs = {}
for name, d in models_data.items():
    vp = d['val_pred']
    tp = d['tes_pred']
    # Determine threshold: hinge models use 0 as threshold, quantum uses 0.5
    if np.abs(vp).max() > 2:  # hinge-style
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
bars_val = ax.bar(x - w/2, [val_accs[m] for m in sorted_models], w, label='Validation', color='steelblue', alpha=0.8)
bars_tes = ax.bar(x + w/2, [tes_accs[m] for m in sorted_models], w, label='Test', color='coral', alpha=0.8)
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

# ── Normalize predictions (min-max on val, apply to test) ──
norm_val = {}
norm_tes = {}
for name, d in models_data.items():
    lo, hi = d['val_pred'].min(), d['val_pred'].max()
    if hi - lo < 1e-10:
        norm_val[name] = np.zeros_like(d['val_pred'])
        norm_tes[name] = np.zeros_like(d['tes_pred'])
    else:
        norm_val[name] = (d['val_pred'] - lo) / (hi - lo)
        norm_tes[name] = (d['tes_pred'] - lo) / (hi - lo)

# ── Ranks ──
ranks_tes = {}
for name in model_names:
    order = norm_tes[name].argsort()
    r = np.empty_like(order)
    r[order] = np.arange(len(order), 0, -1)
    ranks_tes[name] = r

# ── PLOT 2: Rank-Score Graph (CFA-style) ──
fig, ax = plt.subplots(figsize=(10, 6))
# Pick top 7 models for clarity
top7 = sorted(model_names, key=lambda m: tes_accs[m], reverse=True)[:7]
markers = ['o', 's', '^', 'D', 'v', 'P', '*']
colors = plt.cm.tab10(np.linspace(0, 1, 7))
interval = max(1, len(tes_gt) // 50)

for i, name in enumerate(top7):
    r = ranks_tes[name]
    s = norm_tes[name]
    # Sort by rank for plotting
    idx = np.argsort(r)
    r_sorted = r[idx]
    s_sorted = s[idx]
    # Subsample
    sel = np.arange(0, len(r_sorted), interval)
    ax.scatter(r_sorted[sel], s_sorted[sel], label=name[:15], marker=markers[i],
               s=30, alpha=0.7, color=colors[i])
    ax.plot(r_sorted[sel], s_sorted[sel], alpha=0.3, color=colors[i])

ax.set_xlabel('Rank')
ax.set_ylabel('Normalized Prediction Score')
ax.set_title('Rank-Score Graph (Test Dataset, Top 7 Models)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'rank_score_graph.pdf'), bbox_inches='tight')
plt.close()
print('Saved: rank_score_graph.pdf')

# ── Cognitive Diversity ──
def cognitive_diversity(norm_a, ranks_a, norm_b, ranks_b):
    n = len(norm_a)
    scores_a = {r: s for r, s in zip(ranks_a, norm_a)}
    scores_b = {r: s for r, s in zip(ranks_b, norm_b)}
    div_sum = sum((scores_a.get(r, 0) - scores_b.get(r, 0))**2 for r in scores_a)
    return sqrt(div_sum / n) if n > 0 else 0

cd_matrix = np.zeros((len(top7), len(top7)))
for i, m1 in enumerate(top7):
    for j, m2 in enumerate(top7):
        if i != j:
            cd_matrix[i, j] = cognitive_diversity(
                norm_tes[m1], ranks_tes[m1], norm_tes[m2], ranks_tes[m2])

# ── PLOT 3: Cognitive Diversity Heatmap ──
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cd_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(top7)))
ax.set_yticks(range(len(top7)))
ax.set_xticklabels([m[:12] for m in top7], rotation=45, ha='right', fontsize=8)
ax.set_yticklabels([m[:12] for m in top7], fontsize=8)
for i in range(len(top7)):
    for j in range(len(top7)):
        if i != j:
            ax.text(j, i, f'{cd_matrix[i,j]:.3f}', ha='center', va='center', fontsize=7)
ax.set_title('Cognitive Diversity Matrix (Test Dataset)')
plt.colorbar(im, ax=ax, label='Diversity Score')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'diversity_heatmap.pdf'), bbox_inches='tight')
plt.close()
print('Saved: diversity_heatmap.pdf')

# ── PLOT 4: CFA Greedy Results ──
# Full dataset sklearn-only CFA
from pred_lstm import _load_predictions, _cfa_eval_combo
from models import cfa as unified_cfa

data = _load_predictions(RUN_DIR, summary=None)
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

# Evaluate all 6 methods for top individual models and key combinations
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
            break  # just best method per combo

# Bar chart of best CFA combinations
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

# Loss
axes[0].plot(epochs, losses, 'o-', color='darkblue', linewidth=2, markersize=8)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Loss (MSE)')
axes[0].set_title('Quantum LSTM Training Loss')
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0.248, 0.253)

# Accuracy
axes[1].plot(epochs, val_acc_q, 's-', color='steelblue', linewidth=2, markersize=8, label='Validation')
axes[1].plot(epochs, tes_acc_q, 'D-', color='coral', linewidth=2, markersize=8, label='Test')
axes[1].axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Quantum LSTM Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0.44, 0.54)

# Epoch time
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

# ── PLOT 6: CFA Method Comparison (6 methods on key combos) ──
# Evaluate all 6 methods for the top 3 combinations
eval_combos = [
    ['GRADIENT BOOST', 'EXTRA TREES'],
    ['SGD MODHUBER', 'PASSIVE AGGRESSIVE', 'BAGGING DT2'],
    ['GRADIENT BOOST', 'QDA', 'SGD MODHUBER'],
]

method_results = []
for combo in eval_combos:
    if not all(c in usable for c in combo):
        continue
    subset_norm_val = norm_val_df[combo]
    subset_norm_tes = norm_tes_df[combo]
    for method_name in methods:
        try:
            val_s = unified_cfa.fuse(subset_norm_val, val_gt_cfa, method_name, ds, perf)
            val_acc = unified_cfa.evaluate(val_s, val_gt_cfa, 'accuracy')
            tes_s = unified_cfa.fuse(subset_norm_tes, tes_gt_cfa, method_name, ds, perf)
            tes_acc = unified_cfa.evaluate(tes_s, tes_gt_cfa, 'accuracy')
            method_results.append({
                'combo': '+'.join([c[:6] for c in combo]),
                'method': method_name,
                'val': val_acc,
                'test': tes_acc
            })
        except:
            pass

if method_results:
    df_mr = pd.DataFrame(method_results)
    combos_unique = df_mr['combo'].unique()
    methods_unique = df_mr['method'].unique()

    fig, axes = plt.subplots(1, len(combos_unique), figsize=(5*len(combos_unique), 5), sharey=True)
    if len(combos_unique) == 1:
        axes = [axes]

    score_colors = {'ASC': 'steelblue', 'WSCP': 'royalblue', 'WSCDS': 'navy'}
    rank_colors = {'ARC': 'coral', 'WRCP': 'salmon', 'WRCDS': 'darkred'}
    all_colors = {**score_colors, **rank_colors}

    for idx, combo_name in enumerate(combos_unique):
        ax = axes[idx]
        sub = df_mr[df_mr['combo'] == combo_name]
        x = np.arange(len(sub))
        w = 0.35
        colors_v = [all_colors.get(m, 'gray') for m in sub['method']]
        colors_t = [all_colors.get(m, 'gray') for m in sub['method']]
        ax.bar(x - w/2, sub['val'].values, w, color=colors_v, alpha=0.6, label='Val' if idx==0 else '')
        ax.bar(x + w/2, sub['test'].values, w, color=colors_t, alpha=0.9, edgecolor='black', linewidth=0.5, label='Test' if idx==0 else '')
        ax.set_xticks(x)
        ax.set_xticklabels(sub['method'].values, rotation=45, ha='right', fontsize=8)
        ax.set_title(combo_name, fontsize=10)
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0.48, 0.60)
        if idx == 0:
            ax.set_ylabel('Accuracy')

    fig.suptitle('CFA Fusion Methods Comparison (Score=Blue, Rank=Red)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'cfa_methods_comparison.pdf'), bbox_inches='tight')
    plt.close()
    print('Saved: cfa_methods_comparison.pdf')

print('\nAll plots generated in:', PLOT_DIR)
