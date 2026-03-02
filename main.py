"""
Image Classification Pipeline with Combinatorial Fusion Analysis (CFA).

Trains models on MNIST (and other datasets), saves predictions to a
timestamped run directory, then runs CFA to find the best ensemble.

--- Actions ---
  train    Train model(s), save predictions, then auto-run CFA.
  cfa      Run CFA on a previous run's saved predictions (no retraining).

--- Model Flags (-m) ---
  rf       Random Forest only
  cnn      CNN only
  all      All auto-discovered models (default)

--- Usage Examples ---
  # Train Random Forest + CNN on MNIST, then auto-run CFA
  uv run --python .venv/bin/python python main.py -o train -m all

  # Train only CNN for 10 epochs
  uv run --python .venv/bin/python python main.py -o train -m cnn --epochs 10

  # Train only Random Forest
  uv run --python .venv/bin/python python main.py -o train -m rf

  # Run CFA on the most recent run (no retraining)
  uv run --python .venv/bin/python python main.py -o cfa

  # Run CFA on a specific run directory
  uv run --python .venv/bin/python python main.py -o cfa --run_dir data/run_20260302_120000

  # Use AUROC instead of accuracy as the CFA metric
  uv run --python .venv/bin/python python main.py -o train -m all --cfa_metric auroc
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import cfa as cfa_module
from data import load_dataset
from models import get_models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_latest_run(data_dir: str) -> str:
    """
    Return the path of the most recently created run_* directory.
    Used when --run_dir is not specified for CFA.
    """
    runs = sorted(glob.glob(os.path.join(data_dir, "run_*")))
    if not runs:
        sys.exit(f"No run directories found in '{data_dir}/'. Run -o train first.")
    return runs[-1]


def save_predictions(run_dir: str, model_name: str, train_probs: np.ndarray, test_probs: np.ndarray):
    """
    Save (N, K) probability arrays for a model as .npy files.

    .npy format is compact and fast to load; human-readable via np.load().
    Files are named: pred_{MODEL_NAME}_train.npy and pred_{MODEL_NAME}_test.npy
    """
    np.save(os.path.join(run_dir, f"pred_{model_name}_train.npy"), train_probs)
    np.save(os.path.join(run_dir, f"pred_{model_name}_test.npy"),  test_probs)


def load_predictions(run_dir: str) -> tuple[dict, dict]:
    """
    Load all saved prediction files from a run directory.

    Returns two dicts: {model_name: (N, K) array} for train and test sets.
    """
    train_preds, test_preds = {}, {}

    for fpath in sorted(glob.glob(os.path.join(run_dir, "pred_*_train.npy"))):
        # Extract model name: "pred_CNN_train.npy" → "CNN"
        model_name = Path(fpath).stem.removeprefix("pred_").removesuffix("_train")
        train_preds[model_name] = np.load(fpath)

    for fpath in sorted(glob.glob(os.path.join(run_dir, "pred_*_test.npy"))):
        model_name = Path(fpath).stem.removeprefix("pred_").removesuffix("_test")
        test_preds[model_name] = np.load(fpath)

    return train_preds, test_preds


# ---------------------------------------------------------------------------
# Train action
# ---------------------------------------------------------------------------

def run_train(args, X_train, y_train, X_test, y_test, run_dir: str):
    """
    Train all requested models, evaluate on train + test, save predictions.

    For each model:
      1. Call model.train(X_train, y_train)
      2. Call model.predict_proba() on both train and test sets
      3. Compute train_acc and test_acc (argmax of probability matrix)
      4. Save probability arrays as .npy files
      5. Record results in results.csv

    Models that crash are marked ERROR and excluded from CFA.
    """
    models  = get_models(args.model, args)
    results = []  # rows for results.csv

    for model in models:
        print(f"\n{'='*55}")
        print(f"  Training: {model.name}")
        print(f"{'='*55}")

        t_start = time.time()

        try:
            # Train
            model.train(X_train, y_train)
            t_train = time.time() - t_start

            # Predict probabilities on both splits
            train_probs = model.predict_proba(X_train)  # (N_train, K)
            test_probs  = model.predict_proba(X_test)   # (N_test,  K)

            # Accuracy = fraction where argmax prediction matches true label
            train_acc = float(np.mean(np.argmax(train_probs, axis=1) == y_train))
            test_acc  = float(np.mean(np.argmax(test_probs,  axis=1) == y_test))

            print(f"\n  train_acc={train_acc:.4f}  |  test_acc={test_acc:.4f}  |  time={t_train:.1f}s")

            # Save probability arrays for CFA
            save_predictions(run_dir, model.name, train_probs, test_probs)

            results.append({
                "model":        model.name,
                "train_acc":    round(train_acc, 4),
                "test_acc":     round(test_acc, 4),
                "train_time_s": round(t_train, 2),
                "status":       "DONE",
            })

        except Exception as e:
            t_elapsed = time.time() - t_start
            print(f"\n  ERROR after {t_elapsed:.1f}s: {e}")
            results.append({
                "model":        model.name,
                "train_acc":    0.0,
                "test_acc":     0.0,
                "train_time_s": round(t_elapsed, 2),
                "status":       f"ERROR: {e}",
            })

    # Write per-model summary table
    df = pd.DataFrame(results)
    csv_path = os.path.join(run_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*55}")
    print(f"Results saved to {csv_path}")
    print(df.to_string(index=False))

    return results


# ---------------------------------------------------------------------------
# CFA action
# ---------------------------------------------------------------------------

def run_cfa(args, y_train: np.ndarray, y_test: np.ndarray, run_dir: str):
    """
    Load saved predictions from run_dir and run CFA fusion.

    Steps:
      1. Load all pred_*_train.npy and pred_*_test.npy files
      2. Compute diversity and performance weights from training predictions
      3. Evaluate all 2^N model subsets × 6 fusion methods
      4. Save ranked results to cfa_results.csv
    """
    print(f"\n{'='*55}")
    print(f"  Running CFA on: {run_dir}")
    print(f"  Metric: {args.cfa_metric}")
    print(f"{'='*55}")

    train_preds, test_preds = load_predictions(run_dir)

    if not train_preds:
        sys.exit(f"No prediction files found in '{run_dir}'. Run -o train first.")

    print(f"  Loaded {len(train_preds)} model(s): {list(train_preds.keys())}")

    # Run CFA — returns DataFrame sorted by score descending
    results_df = cfa_module.run_cfa(
        train_preds, test_preds,
        y_train, y_test,
        metric=args.cfa_metric,
    )

    # Save full results table
    out_path = os.path.join(run_dir, "cfa_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  Full CFA results saved to {out_path}")

    # Show top 20 rows (sorted by score)
    print("\n  Top results:")
    print(results_df.head(20).to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Image classification pipeline with CFA ensemble fusion.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Action ---
    parser.add_argument(
        "-o", "--action",
        choices=["train", "cfa"],
        default="train",
        help="Action to perform: 'train' trains models; 'cfa' runs CFA on saved predictions. (default: train)",
    )

    # --- Model selection ---
    parser.add_argument(
        "-m", "--model",
        default="all",
        help="Which models to train: rf, cnn, or all (default: all)",
    )

    # --- Dataset ---
    parser.add_argument(
        "--dataset",
        default="mnist",
        help="Dataset to use: mnist, fashion (default: mnist)",
    )

    # --- CNN hyperparameter ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs for CNN (default: 5). More epochs = better accuracy but slower.",
    )

    # --- CFA options ---
    parser.add_argument(
        "--cfa_metric",
        default="accuracy",
        choices=["accuracy", "auroc"],
        help="Metric for CFA scoring: accuracy (default) or auroc.",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
        help="Run directory for -o cfa. Uses most recent run if not specified.",
    )

    # --- Paths ---
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory for dataset downloads and run outputs (default: data/)",
    )

    args = parser.parse_args()

    # --- Load dataset ---
    print(f"\nLoading dataset: {args.dataset}")
    X_train, y_train, X_test, y_test = load_dataset(args.dataset, args.data_dir)
    n_classes = len(np.unique(y_train))
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}  |  Classes: {n_classes}")

    # --- Dispatch to action ---
    if args.action == "train":
        # Create a new timestamped run directory
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.data_dir, f"run_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"  Run directory: {run_dir}")

        # Save ground truth labels once (used by CFA later)
        np.save(os.path.join(run_dir, "labels_train.npy"), y_train)
        np.save(os.path.join(run_dir, "labels_test.npy"),  y_test)

        # Train and save predictions
        run_train(args, X_train, y_train, X_test, y_test, run_dir)

        # Auto-run CFA immediately after training
        print(f"\n{'='*55}")
        print("  Auto-running CFA on training results...")
        run_cfa(args, y_train, y_test, run_dir)

    elif args.action == "cfa":
        # Find the run directory to use
        run_dir = args.run_dir or get_latest_run(args.data_dir)

        # Load the labels saved at training time
        labels_train_path = os.path.join(run_dir, "labels_train.npy")
        labels_test_path  = os.path.join(run_dir, "labels_test.npy")

        if not os.path.exists(labels_train_path):
            sys.exit(f"Labels not found in '{run_dir}'. Was this directory created by -o train?")

        y_train = np.load(labels_train_path)
        y_test  = np.load(labels_test_path)

        run_cfa(args, y_train, y_test, run_dir)


if __name__ == "__main__":
    main()
