"""
Combinatorial Fusion Analysis (CFA) — unified single-file implementation.

Unifies ADMET, CFA4SA, Wallace, and Horatio implementations.
CD formula: sqrt(sum((f_A - f_B)^2) / n)
Train-based normalization. All 6 fusion methods.

Usage:
    from cfa import cfa_single_layer, compute_cd_matrix

    results = cfa_single_layer(train_preds_df, test_preds_df, y_train, y_test,
                               metric="auroc")
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error,
)
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Core: normalization, RSC, CD, DS
# ---------------------------------------------------------------------------

def normalize_minmax(values, ref_min=None, ref_max=None):
    """Min-max normalize. If ref_min/ref_max given, use those (train-based)."""
    if ref_min is None: ref_min = np.min(values)
    if ref_max is None: ref_max = np.max(values)
    denom = ref_max - ref_min
    return np.zeros_like(values, dtype=float) if denom == 0 else (values - ref_min) / denom


def rank_scores(values):
    """Assign ranks (1 = highest score)."""
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def compute_rsc(scores):
    """Rank-Score Characteristic: normalized scores sorted descending."""
    return np.sort(scores)[::-1]


def cognitive_diversity(rsc_a, rsc_b):
    """CD between two RSC curves: sqrt(sum((a-b)^2) / n)."""
    n = len(rsc_a)
    return 0.0 if n == 0 else np.sqrt(np.sum((rsc_a - rsc_b) ** 2) / n)


def compute_cd_matrix(preds_df):
    """Compute pairwise CD, diversity strength (DS), and RSC for all models.

    Returns: (cd_dict, ds_dict, rsc_dict)
    """
    cols = list(preds_df.columns)
    rsc = {c: compute_rsc(normalize_minmax(preds_df[c].values)) for c in cols}
    cd = {tuple(sorted([a, b])): cognitive_diversity(rsc[a], rsc[b])
          for a, b in itertools.combinations(cols, 2)}
    ds = {c: np.mean([cd[tuple(sorted([c, o]))] for o in cols if o != c]) or 0.0
          for c in cols}
    return cd, ds, rsc

# ---------------------------------------------------------------------------
# Fusion methods (6 total)
# ---------------------------------------------------------------------------

def _weighted_combine(df, weights, use_ranks=False):
    cols = list(df.columns)
    w = np.array([weights[c] for c in cols])
    w = w / w.sum()
    data = df.rank(ascending=False).values if use_ranks else df.values
    return np.average(data, axis=1, weights=w)


def average_score_combination(df, subset):
    """ASC: simple average of normalized scores."""
    return df[list(subset)].mean(axis=1).values

def average_rank_combination(df, subset):
    """ARC: simple average of ranks."""
    return df[list(subset)].rank(ascending=False).mean(axis=1).values

def weighted_score_by_diversity(df, subset, ds):
    """WSCDS: scores weighted by diversity strength."""
    return _weighted_combine(df[list(subset)], {m: ds[m] for m in subset})

def weighted_rank_by_diversity(df, subset, ds):
    """WRCDS: ranks weighted by diversity strength."""
    return _weighted_combine(df[list(subset)], {m: ds[m] for m in subset}, True)

def weighted_score_by_performance(df, subset, perf):
    """WSCP: scores weighted by individual performance."""
    return _weighted_combine(df[list(subset)], {m: perf[m] for m in subset})

def weighted_rank_by_performance(df, subset, perf):
    """WRCP: ranks weighted by individual performance."""
    return _weighted_combine(df[list(subset)], {m: perf[m] for m in subset}, True)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(predictions, y_true, metric, is_rank=False):
    """Evaluate predictions against ground truth.

    Supported metrics: auroc, auprc, spearman, mae, rmse,
    directional_accuracy, accuracy, f1, precision, recall.
    """
    if metric == "auroc":
        return (1 - roc_auc_score(y_true, predictions)) if is_rank else roc_auc_score(y_true, predictions)
    if metric == "auprc":
        return (1 - average_precision_score(y_true, predictions)) if is_rank else average_precision_score(y_true, predictions)
    if metric == "spearman":
        return spearmanr(y_true, predictions)[0]
    if metric == "mae":
        return mean_absolute_error(y_true, predictions)
    if metric == "rmse":
        return np.sqrt(mean_squared_error(y_true, predictions))
    if metric == "directional_accuracy":
        return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(predictions))) if len(y_true) >= 2 else 0.0
    binary = (predictions <= np.median(predictions)).astype(int) if is_rank else (predictions >= 0.5).astype(int)
    if metric == "accuracy":  return accuracy_score(y_true, binary)
    if metric == "f1":        return f1_score(y_true, binary, zero_division=0)
    if metric == "precision": return precision_score(y_true, binary, zero_division=0)
    if metric == "recall":    return recall_score(y_true, binary, zero_division=0)
    raise ValueError(f"Unknown metric: {metric}")

# ---------------------------------------------------------------------------
# Sliding Rule: model selection for large pools
# ---------------------------------------------------------------------------

def sliding_rule_select(train_preds_df, y_train, n_select, metric="auroc"):
    """Select a subset of models balancing performance and diversity.

    Interleaves from performance-ranked and diversity-ranked lists,
    skipping duplicates, until n_select models are chosen.

    Args:
        train_preds_df: DataFrame, columns=model names, rows=train samples.
        y_train: array-like ground truth for train.
        n_select: number of models to select.
        metric: metric for individual performance ranking.

    Returns:
        selected: list of model names.
    """
    models = list(train_preds_df.columns)
    if n_select >= len(models):
        return models

    # Normalize and score each model
    norm = pd.DataFrame(index=train_preds_df.index)
    for c in models:
        norm[c] = normalize_minmax(train_preds_df[c].values)
    perf = {c: evaluate(norm[c].values, y_train, metric) for c in models}
    _, ds, _ = compute_cd_matrix(train_preds_df)

    perf_ranked = sorted(models, key=lambda m: perf[m], reverse=True)
    div_ranked = sorted(models, key=lambda m: ds[m], reverse=True)

    selected, seen = [], set()
    pi, di, pick_perf = 0, 0, True
    while len(selected) < n_select:
        if pick_perf:
            while pi < len(perf_ranked):
                c = perf_ranked[pi]; pi += 1
                if c not in seen:
                    selected.append(c); seen.add(c); break
        else:
            while di < len(div_ranked):
                c = div_ranked[di]; di += 1
                if c not in seen:
                    selected.append(c); seen.add(c); break
        pick_perf = not pick_perf
        if pi >= len(perf_ranked) and di >= len(div_ranked):
            break
    return selected

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

ALL_METHODS = ["ASC", "ARC", "WSCDS", "WRCDS", "WSCP", "WRCP"]

def cfa_single_layer(train_preds_df, test_preds_df, y_train, y_test,
                     metric="auroc", methods=None):
    """Run CFA over all model combinations and fusion methods.

    Args:
        train_preds_df: DataFrame, columns=model names, rows=train samples.
        test_preds_df:  DataFrame, columns=model names, rows=test samples.
        y_train: array-like ground truth for train.
        y_test:  array-like ground truth for test.
        metric:  one of auroc, auprc, accuracy, f1, precision, recall,
                 spearman, mae, rmse, directional_accuracy.
        methods: list of fusion methods to use (default: all 6).

    Returns:
        DataFrame with columns [combination, method, n_models, score].
    """
    if methods is None:
        methods = ALL_METHODS
    models = list(train_preds_df.columns)

    # Train-based normalization
    norm_test = pd.DataFrame(index=test_preds_df.index)
    norm_train = pd.DataFrame(index=train_preds_df.index)
    for c in models:
        lo, hi = train_preds_df[c].min(), train_preds_df[c].max()
        norm_test[c] = normalize_minmax(test_preds_df[c].values, lo, hi)
        norm_train[c] = normalize_minmax(train_preds_df[c].values, lo, hi)

    _, ds, _ = compute_cd_matrix(train_preds_df)
    perf = {c: evaluate(norm_train[c].values, y_train, metric) for c in models}

    results = []
    for r in range(1, len(models) + 1):
        for subset in itertools.combinations(models, r):
            name = "+".join(subset)
            if r == 1:
                results.append({"combination": name, "method": "individual",
                    "n_models": 1, "score": evaluate(norm_test[subset[0]].values, y_test, metric)})
                continue
            fn = {
                "ASC":   lambda s=subset: average_score_combination(norm_test, s),
                "ARC":   lambda s=subset: average_rank_combination(norm_test, s),
                "WSCDS": lambda s=subset: weighted_score_by_diversity(norm_test, s, ds),
                "WRCDS": lambda s=subset: weighted_rank_by_diversity(norm_test, s, ds),
                "WSCP":  lambda s=subset: weighted_score_by_performance(norm_test, s, perf),
                "WRCP":  lambda s=subset: weighted_rank_by_performance(norm_test, s, perf),
            }
            for m in methods:
                if m not in fn:
                    continue
                is_rank = m in ("ARC", "WRCDS", "WRCP")
                results.append({"combination": name, "method": m, "n_models": r,
                    "score": evaluate(fn[m](), y_test, metric, is_rank)})

    return pd.DataFrame(results)
