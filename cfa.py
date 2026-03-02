"""
Combinatorial Fusion Analysis (CFA) for multiclass image classification.

Adapted from the quantumstd1 archive (archive/models/cfa.py).
Original CFA theory: Jiang et al., Rank-Score Characteristic (RSC) functions.

Each model outputs an (N, K) probability matrix (K = number of classes).
CFA fuses these matrices using 6 methods and finds which combination
of models produces the best ensemble accuracy.

--- CFA Theory ---
The key insight: models with DIVERSE errors improve more when combined than
models with CORRELATED errors. Even a weaker model can help the ensemble
if its mistakes don't overlap with the other models' mistakes.

Diversity is measured via Cognitive Diversity (CD):
  CD(A, B) = sqrt( mean( (RSC_A - RSC_B)^2 ) )
  where RSC is the Rank-Score Characteristic curve (confidence scores sorted
  descending). Low CD = similar predictions, High CD = complementary.

Diversity Strength (DS) for model A = mean CD against all other models.

--- 6 Fusion Methods ---
  ASC:   Average Score Combination  — simple mean of probability matrices
  ARC:   Average Rank Combination   — Borda count: average per-sample class rankings
  WSCDS: Weighted Score by Diversity Strength — high-diversity models get more weight
  WRCDS: Weighted Rank by Diversity Strength  — low-diversity (consensus) models get more weight
  WSCP:  Weighted Score by Performance        — higher-accuracy models get more weight
  WRCP:  Weighted Rank by Performance         — lower-accuracy models get more weight (for rank)

--- Multiclass Adaptation ---
Binary CFA fuses single scores per sample. For K classes:
  Score fusion:  average (weighted) the (N, K) probability matrices → argmax → predicted class
  Rank fusion:   for each sample, rank classes by probability (1=best); average ranks → argmin
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------------------------
# Core: RSC, Cognitive Diversity, Diversity Strength, Performance
# ---------------------------------------------------------------------------

def _confidence(probs: np.ndarray) -> np.ndarray:
    """
    Per-sample confidence: max probability across classes.
    Shape: (N,). Used to build RSC curves for diversity computation.
    """
    return probs.max(axis=1)


def _normalize_minmax(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1D array to [0, 1]. Returns zeros if all values equal."""
    lo, hi = values.min(), values.max()
    return np.zeros_like(values, dtype=float) if hi == lo else (values - lo) / (hi - lo)


def _compute_rsc(probs: np.ndarray) -> np.ndarray:
    """
    Rank-Score Characteristic (RSC) curve for a model.

    Computes normalized confidence scores sorted in descending order.
    Two models with similar RSC curves have low CD (correlated predictions).
    Shape: (N,).
    """
    conf = _normalize_minmax(_confidence(probs))
    return np.sort(conf)[::-1]  # descending


def _cognitive_diversity(rsc_a: np.ndarray, rsc_b: np.ndarray) -> float:
    """
    Cognitive Diversity between two RSC curves.

    CD(A, B) = sqrt( sum((RSC_A - RSC_B)^2) / n )

    Higher CD = more complementary models (different error patterns).
    Lower CD = redundant models (similar error patterns).
    """
    n = min(len(rsc_a), len(rsc_b))
    return float(np.sqrt(np.sum((rsc_a[:n] - rsc_b[:n]) ** 2) / n))


def compute_diversity_strength(train_preds: dict) -> dict:
    """
    Compute Diversity Strength for each model (mean CD against all others).

    Higher DS = model is more complementary to the pool = more valuable in ensemble.

    Args:
        train_preds: {model_name: (N, K) probability array on training set}

    Returns:
        {model_name: diversity_strength (float)}
    """
    models = list(train_preds.keys())
    # Pre-compute RSC for all models
    rscs = {m: _compute_rsc(train_preds[m]) for m in models}

    ds = {}
    for m in models:
        others = [o for o in models if o != m]
        if not others:
            ds[m] = 1.0  # single model has no diversity to measure
        else:
            # Mean CD against all other models
            ds[m] = float(np.mean([_cognitive_diversity(rscs[m], rscs[o]) for o in others]))
    return ds


def compute_performance_strength(train_preds: dict, y_train: np.ndarray) -> dict:
    """
    Compute Performance Strength for each model = training accuracy.

    Used to weight models by individual quality (WSCP/WRCP methods).

    Args:
        train_preds: {model_name: (N, K) probability array on training set}
        y_train:     (N,) ground truth class labels

    Returns:
        {model_name: accuracy (float in [0, 1])}
    """
    return {
        m: float(accuracy_score(y_train, np.argmax(p, axis=1)))
        for m, p in train_preds.items()
    }


# ---------------------------------------------------------------------------
# Score-based fusion (operates on probability matrices directly)
# ---------------------------------------------------------------------------

def fuse_score(probs_list: list, weights: list = None) -> np.ndarray:
    """
    Weighted (or uniform) average of (N, K) probability matrices.

    For ASC: weights=None → simple mean across models
    For WSCDS/WSCP: weights = diversity or performance values

    Returns: (N, K) fused probability matrix (can be used with argmax)
    """
    if weights is None:
        # Uniform average: sum then divide by count
        return np.mean(probs_list, axis=0)

    # Weighted average: normalize weights to sum to 1
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return sum(p * wi for p, wi in zip(probs_list, w))


# ---------------------------------------------------------------------------
# Rank-based fusion (Borda count for multiclass)
# ---------------------------------------------------------------------------

def _probs_to_ranks(p: np.ndarray) -> np.ndarray:
    """
    Convert (N, K) probability matrix to within-sample class rankings.

    For each sample i, rank its K class probabilities (rank 1 = highest prob).
    This is a Borda count: class with rank 1 is the model's top prediction.

    Returns: (N, K) rank matrix (int), values in [1, K]
    """
    N, K = p.shape
    ranks = np.zeros_like(p, dtype=float)
    for i in range(N):
        # argsort(-p[i]) gives class indices from highest to lowest prob
        order = np.argsort(-p[i])
        # Assign ranks: the class at position 0 in order gets rank 1, etc.
        ranks[i, order] = np.arange(1, K + 1)
    return ranks


def fuse_rank(probs_list: list, weights: list = None) -> np.ndarray:
    """
    Average within-sample class rankings (Borda count fusion).

    For ARC: weights=None → equal Borda count vote
    For WRCDS/WRCP: weights = 1/diversity or 1/performance (low-diversity / low-perf
                              models given more weight to balance consensus)

    Final prediction: argmin of mean rank matrix (lowest average rank = most voted class).

    Returns: (N, K) mean rank matrix (lower = more voted)
    """
    rank_matrices = [_probs_to_ranks(p) for p in probs_list]

    if weights is None:
        return np.mean(rank_matrices, axis=0)

    w = np.array(weights, dtype=float)
    w = w / w.sum()
    return sum(r * wi for r, wi in zip(rank_matrices, w))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(fused: np.ndarray, y_true: np.ndarray, is_rank: bool, metric: str) -> float:
    """
    Evaluate a fused (N, K) matrix against ground truth.

    For score matrices: argmax gives predicted class.
    For rank matrices:  argmin gives predicted class (lowest rank = best class).

    Supported metrics:
        accuracy — fraction of correctly predicted classes
        auroc    — macro one-vs-rest AUROC (requires probability scores, not ranks)
    """
    preds = np.argmin(fused, axis=1) if is_rank else np.argmax(fused, axis=1)

    if metric == "accuracy":
        return float(accuracy_score(y_true, preds))

    if metric == "auroc":
        # OvR AUROC: treat each class as a binary problem, then average
        probs = -fused if is_rank else fused  # invert ranks so higher = better
        try:
            return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except ValueError:
            return 0.0

    raise ValueError(f"Unknown metric '{metric}'. Choose 'accuracy' or 'auroc'.")


# ---------------------------------------------------------------------------
# Main CFA pipeline
# ---------------------------------------------------------------------------

def run_cfa(
    train_preds: dict,
    test_preds: dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """
    Run CFA over all model subsets and all 6 fusion methods.

    Scores individual models first, then every combination of 2+ models.
    Weights for diversity/performance methods are computed on the training set
    (to avoid using test labels for model selection).

    Args:
        train_preds: {model_name: (N_train, K) probability array}
        test_preds:  {model_name: (N_test,  K) probability array}
        y_train:     (N_train,) ground truth labels for training set
        y_test:      (N_test,)  ground truth labels for test set
        metric:      "accuracy" or "auroc"

    Returns:
        DataFrame with columns [combination, method, n_models, score],
        sorted by score descending. Saves best individual and best fusion.
    """
    models = list(train_preds.keys())

    # Compute weights from training predictions only (no test data leakage)
    ds   = compute_diversity_strength(train_preds)
    perf = compute_performance_strength(train_preds, y_train)

    print(f"\n  Diversity Strength:   { {m: f'{v:.4f}' for m, v in ds.items()} }")
    print(f"  Performance Strength: { {m: f'{v:.4f}' for m, v in perf.items()} }")

    rows = []

    # --- Score each model individually ---
    for m in models:
        score = _evaluate(test_preds[m], y_test, is_rank=False, metric=metric)
        rows.append({"combination": m, "method": "individual", "n_models": 1, "score": score})

    # --- Score all combinations of 2+ models with all 6 fusion methods ---
    for r in range(2, len(models) + 1):
        for subset in itertools.combinations(models, r):
            name = "+".join(subset)

            # Gather probability matrices for this subset
            p_train = [train_preds[m] for m in subset]
            p_test  = [test_preds[m]  for m in subset]

            # Weights for diversity-based methods
            ds_w    = [ds[m]   for m in subset]           # ASC/ARC: not used
            # Inverse diversity: low-diversity (consensus) models dominate rank combo
            ds_inv  = [1.0 / max(ds[m], 1e-8) for m in subset]

            # Weights for performance-based methods
            perf_w  = [perf[m] for m in subset]           # WSCP: high-perf = more weight
            # Inverse performance: used by WRCP (balance strong models' rank influence)
            perf_inv = [1.0 / max(perf[m], 1e-8) for m in subset]

            # Define all 6 fusion combinations:
            #   (method_name, fused_matrix, is_rank_based)
            combos = [
                ("ASC",   fuse_score(p_test),            False),  # uniform score average
                ("ARC",   fuse_rank(p_test),              True),  # uniform Borda count
                ("WSCDS", fuse_score(p_test, ds_w),      False),  # diversity-weighted score
                ("WRCDS", fuse_rank(p_test,  ds_inv),    True),   # consensus-weighted rank
                ("WSCP",  fuse_score(p_test, perf_w),    False),  # performance-weighted score
                ("WRCP",  fuse_rank(p_test,  perf_inv),  True),   # balanced-perf rank
            ]

            for method_name, fused, is_rank in combos:
                score = _evaluate(fused, y_test, is_rank, metric)
                rows.append({
                    "combination": name,
                    "method":      method_name,
                    "n_models":    r,
                    "score":       score,
                })

    # Build results DataFrame, sorted by score descending
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    # Print summary: best individual vs best fusion
    best_ind = df[df["method"] == "individual"].iloc[0]
    print(f"\n  Best individual: {best_ind['combination']}"
          f"  ({metric}={best_ind['score']:.4f})")

    # Fusion results only exist when 2+ models were evaluated
    fusion_rows = df[df["method"] != "individual"]
    if fusion_rows.empty:
        print("  (Need 2+ models for fusion. Train more models and re-run CFA.)")
    else:
        best_fus = fusion_rows.iloc[0]
        print(f"  Best fusion:     {best_fus['combination']}"
              f"  [{best_fus['method']}]"
              f"  ({metric}={best_fus['score']:.4f})")

        improvement = best_fus["score"] - best_ind["score"]
        print(f"  Improvement: {improvement:+.4f}  ({improvement * 100:+.2f}%)")

        if improvement > 0:
            print("  >>> CFA ensemble outperforms best individual model! <<<")
        else:
            print("  >>> Best individual model holds (no CFA improvement on this run) <<<")

    return df
