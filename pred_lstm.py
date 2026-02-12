import argparse
from contextlib import nullcontext as _nullcontext
import csv
import ctypes
from datetime import datetime
import gc
import io
import itertools
import numpy as np
import os
import pathlib
import psutil
import signal
import threading
from time import time

os.environ['TF_USE_LEGACY_KERAS'] = '1'
# Auto-detect pip-installed NVIDIA CUDA libs for GPU support
_nvidia_dir = pathlib.Path(__file__).resolve().parent / '.venv' / 'lib'
for _lib in _nvidia_dir.rglob('nvidia/*/lib'):
    if _lib.is_dir():
        os.environ['LD_LIBRARY_PATH'] = str(_lib) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

from models import cfa as unified_cfa

import pandas as pd
from rich.console import Console
from rich.table import Table

from models import (AWLSTM, QuantumTrainer, OOMTestTrainer,
                     RandomForestTrainer, GradientBoostTrainer, MLPTrainer,
                     LogisticRegressionTrainer)
from models import (
    AWLSTM, QuantumTrainer, OOMTestTrainer,
    RandomForestTrainer, GradientBoostTrainer, MLPTrainer,
    LogisticRegressionTrainer,
    SGDLogTrainer, SGDHingeTrainer, PassiveAggressiveTrainer, RidgeTrainer,
    PerceptronTrainer, LDATrainer, QDATrainer, GaussianNBTrainer,
    NearestCentroidTrainer, DecisionTreeTrainer, ExtraTreesTrainer,
    AdaBoostTrainer, LinearSVCTrainer, HistGBDTTrainer, BaggingDTTrainer, DummyMostFreqTrainer, DummyStratifiedTrainer, KNN3Trainer, KNN11DistTrainer, SGDModHuberTrainer, GaussianNB1e8Trainer, GaussianNB1e7Trainer,
)



# ---------------------------------------------------------------------------
# RunSummary — pipeline table + CSV results + prediction saving
# ---------------------------------------------------------------------------

class RunSummary:
    """Shows a table of all models to train, updated with results as each finishes."""

    def __init__(self):
        self._console = Console()
        self._models = []
        self._timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join('data', 'run_%s' % self._timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        self._csv_path = os.path.join(self.run_dir, 'results.csv')

    def add_model(self, name, epochs):
        self._models.append({
            'name': name, 'status': 'PENDING', 'epochs': epochs,
            'val_acc': '-', 'val_mcc': '-',
            'test_acc': '-', 'test_mcc': '-', 'time': '-',
        })

    def start_model(self, name):
        for m in self._models:
            if m['name'] == name:
                m['status'] = 'TRAINING'

    def finish_model(self, name, val_perf, test_perf, elapsed):
        for m in self._models:
            if m['name'] == name:
                m['status'] = 'DONE'
                m['val_acc'] = '%.4f' % val_perf.get('acc', 0)
                m['val_mcc'] = '%.4f' % val_perf.get('mcc', 0)
                m['test_acc'] = '%.4f' % test_perf.get('acc', 0)
                m['test_mcc'] = '%.4f' % test_perf.get('mcc', 0)
                m['time'] = '%.1fs' % elapsed
        self._save_csv()

    def predictions_path(self, model_name, split):
        safe_name = model_name.replace(' ', '_')
        return os.path.join(self.run_dir, 'pred_%s_%s.csv' % (safe_name, split))

    def save_predictions(self, model_name, split, data):
        """Save predictions to CSV, splitting into numbered files if over 99 MB."""
        path = self.predictions_path(model_name, split)
        header = 'epoch,prediction,ground_truth'
        max_bytes = 99 * 1024 * 1024

        # Estimate total size from a single row
        buf = io.BytesIO()
        np.savetxt(buf, data[:1], delimiter=',', header=header, comments='')
        header_bytes = len(header.encode()) + 1
        row_bytes = buf.tell() - header_bytes
        total_est = header_bytes + row_bytes * data.shape[0]

        if total_est <= max_bytes:
            np.savetxt(path, data, delimiter=',', header=header, comments='')
            return

        # Split into chunks
        rows_per_file = max(int((max_bytes - header_bytes) / row_bytes), 1)
        base, ext = os.path.splitext(path)
        for idx, start in enumerate(range(0, data.shape[0], rows_per_file), 1):
            chunk_path = '%s_%03d%s' % (base, idx, ext)
            np.savetxt(chunk_path, data[start:start + rows_per_file],
                       delimiter=',', header=header, comments='')

    def _save_csv(self):
        header = ['model', 'status', 'epochs', 'val_acc', 'val_mcc',
                  'test_acc', 'test_mcc', 'time']
        with open(self._csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for m in self._models:
                w.writerow([m['name'], m['status'], m['epochs'],
                            m['val_acc'], m['val_mcc'],
                            m['test_acc'], m['test_mcc'], m['time']])

    def build_table(self):
        table = Table(title='Training Pipeline', expand=True)
        table.add_column('Model', justify='left')
        table.add_column('Status', justify='center')
        table.add_column('Epochs', justify='center')
        table.add_column('Val Acc', justify='center')
        table.add_column('Val MCC', justify='center')
        table.add_column('Test Acc', justify='center')
        table.add_column('Test MCC', justify='center')
        table.add_column('Time', justify='center')
        for m in self._models:
            style = None
            if m['status'] == 'TRAINING':
                style = 'bold yellow'
            elif m['status'] == 'DONE':
                style = 'bold green'
            table.add_row(
                m['name'], m['status'], str(m['epochs']),
                m['val_acc'], m['val_mcc'],
                m['test_acc'], m['test_mcc'], m['time'],
                style=style,
            )
        return table

    def print(self):
        self._console.print(self.build_table())


# ---------------------------------------------------------------------------
# MemoryGuard — per-model RAM limit
# ---------------------------------------------------------------------------

class MemoryGuard:
    """Enforces a per-model RAM limit. Kills the model if it exceeds it."""

    def __init__(self, limit_gb=1.0, check_interval=0.25):
        self.limit_bytes = int(limit_gb * 1024 ** 3)
        self.limit_gb = limit_gb
        self.check_interval = check_interval
        self._proc = psutil.Process()
        self._baseline = 0
        self._peak = 0
        self._exceeded = False
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        gc.collect()
        self._baseline = self._proc.memory_info().rss
        self._peak = self._baseline
        self._exceeded = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        print(f'[MemoryGuard] Baseline RSS: {self._baseline / 1024**2:.0f} MB, '
              f'limit: +{self.limit_gb:.1f} GB')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        used_mb = (self._peak - self._baseline) / 1024 ** 2
        print(f'[MemoryGuard] Peak usage above baseline: {used_mb:.0f} MB')
        gc.collect()
        return False

    def _watch(self):
        main_tid = threading.main_thread().ident
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            used = rss - self._baseline
            if used > self.limit_bytes:
                self._exceeded = True
                print(f'\n[MemoryGuard] KILLED: model used '
                      f'{used / 1024**3:.2f} GB (limit: {self.limit_gb:.1f} GB)')
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(main_tid),
                    ctypes.py_object(MemoryError),
                )
                break
            self._stop.wait(self.check_interval)

    @property
    def exceeded(self):
        return self._exceeded


# ---------------------------------------------------------------------------
# TimeGuard — per-model time limit
# ---------------------------------------------------------------------------

class TimeGuard:
    """Enforces a per-model time limit. Kills the model if it runs too long."""

    def __init__(self, limit_sec=1.0, check_interval=0.25):
        self.limit_sec = limit_sec
        self.check_interval = check_interval
        self._start = 0
        self._exceeded = False
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._start = time()
        self._exceeded = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        print(f'[TimeGuard] Limit: {self.limit_sec:.1f}s')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        elapsed = time() - self._start
        print(f'[TimeGuard] Elapsed: {elapsed:.1f}s')
        return False

    def _watch(self):
        main_tid = threading.main_thread().ident
        while not self._stop.is_set():
            elapsed = time() - self._start
            if elapsed > self.limit_sec:
                self._exceeded = True
                print(f'\n[TimeGuard] KILLED: model ran {elapsed:.1f}s '
                      f'(limit: {self.limit_sec:.1f}s)')
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(main_tid),
                    ctypes.py_object(TimeoutError),
                )
                break
            self._stop.wait(self.check_interval)

    @property
    def exceeded(self):
        return self._exceeded


# ---------------------------------------------------------------------------
# CFA combination of model predictions
# ---------------------------------------------------------------------------

def _load_predictions(run_dir, summary=None):
    """Discover and load prediction CSVs from a run directory.

    Works with a RunSummary (skips killed models) or standalone (scans files).
    Returns (val_preds, tes_preds, val_gt, tes_gt, usable) or None if < 2 models.
    """
    import glob as globmod

    val_files = {}
    tes_files = {}

    if summary:
        for m in summary._models:
            name = m['name']
            if m['status'] != 'DONE' or m['val_acc'] == '0.0000':
                continue
            val_path = summary.predictions_path(name, 'val')
            tes_path = summary.predictions_path(name, 'test')
            if os.path.exists(val_path) and os.path.exists(tes_path):
                val_files[name] = val_path
                tes_files[name] = tes_path
    else:
        # Standalone: scan for pred_*_val.csv / pred_*_test.csv
        for vp in sorted(globmod.glob(os.path.join(run_dir, 'pred_*_val.csv'))):
            base = os.path.basename(vp)
            name = base.replace('pred_', '').replace('_val.csv', '').replace('_', ' ')
            tp = vp.replace('_val.csv', '_test.csv')
            if os.path.exists(tp):
                val_files[name] = vp
                tes_files[name] = tp

    if len(val_files) < 2:
        print('[CFA] Need at least 2 models with predictions, got %d — skipping'
              % len(val_files))
        return None

    val_preds = {}
    tes_preds = {}
    val_gts = {}
    tes_gts = {}
    for name in val_files:
        vd = np.loadtxt(val_files[name], delimiter=',', skiprows=1)
        td = np.loadtxt(tes_files[name], delimiter=',', skiprows=1)
        last_epoch = int(vd[:, 0].max())
        vd = vd[vd[:, 0] == last_epoch]
        td = td[td[:, 0] == last_epoch]
        val_preds[name] = vd[:, 1]
        tes_preds[name] = td[:, 1]
        val_gts[name] = vd[:, 2]
        tes_gts[name] = td[:, 2]

    n_val = min(len(v) for v in val_preds.values())
    n_tes = min(len(v) for v in tes_preds.values())
    usable = list(val_preds.keys())
    for name in usable:
        val_preds[name] = val_preds[name][:n_val]
        tes_preds[name] = tes_preds[name][:n_tes]
    val_gt = (val_gts[usable[0]][:n_val] > 0.5).astype(int)
    tes_gt = (tes_gts[usable[0]][:n_tes] > 0.5).astype(int)

    if len(usable) < 2:
        print('[CFA] Need at least 2 models with predictions, got %d — skipping'
              % len(usable))
        return None

    return val_preds, tes_preds, val_gt, tes_gt, usable


def run_cfa(run_dir, summary=None, time_limit=1.0, models_filter=None):
    """Run CFA combination on model predictions.

    Args:
        run_dir: path to run directory containing prediction CSVs.
        summary: optional RunSummary (used to skip killed models).
        time_limit: max seconds for CFA evaluation (0=unlimited).
        models_filter: optional list of model keys to include (None=all).
    """
    console = Console()
    data = _load_predictions(run_dir, summary)
    if data is None:
        return
    val_preds, tes_preds, val_gt, tes_gt, usable = data

    # Filter to requested models
    if models_filter:
        usable = [m for m in usable if m in models_filter]
        if len(usable) < 2:
            print('[CFA] Need at least 2 models after filter, got %d — skipping'
                  % len(usable))
            return

    print('\n===== CFA COMBINATION =====')
    print('[CFA] Combining %d models: %s' % (len(usable), ', '.join(usable)))
    total_combos = 2 ** len(usable) - 1
    print('[CFA] %d possible combinations, time limit: %s' % (
        total_combos, '%.1fs' % time_limit if time_limit > 0 else 'unlimited'))

    val_df = pd.DataFrame({name: val_preds[name] for name in usable})
    tes_df = pd.DataFrame({name: tes_preds[name] for name in usable})

    # Pre-compute normalization and CD (fast, one-time)
    norm_tes = pd.DataFrame(index=tes_df.index)
    norm_val = pd.DataFrame(index=val_df.index)
    for c in usable:
        lo, hi = val_df[c].min(), val_df[c].max()
        norm_tes[c] = unified_cfa.normalize_minmax(tes_df[c].values, lo, hi)
        norm_val[c] = unified_cfa.normalize_minmax(val_df[c].values, lo, hi)

    _, ds, _ = unified_cfa.compute_cd_matrix(val_df)
    perf = {c: unified_cfa.evaluate(norm_val[c].values, val_gt, 'accuracy')
            for c in usable}

    methods = unified_cfa.ALL_METHODS
    results = []
    combos_done = 0
    timed_out = False
    t0 = time()

    for r in range(1, len(usable) + 1):
        for subset in itertools.combinations(usable, r):
            name = '+'.join(subset)
            if r == 1:
                results.append({'combination': name, 'method': 'individual',
                    'n_models': 1,
                    'score': unified_cfa.evaluate(
                        norm_tes[subset[0]].values, tes_gt, 'accuracy')})
            else:
                fn = {
                    'ASC':   lambda s=subset: unified_cfa.average_score_combination(norm_tes, s),
                    'ARC':   lambda s=subset: unified_cfa.average_rank_combination(norm_tes, s),
                    'WSCDS': lambda s=subset: unified_cfa.weighted_score_by_diversity(norm_tes, s, ds),
                    'WRCDS': lambda s=subset: unified_cfa.weighted_rank_by_diversity(norm_tes, s, ds),
                    'WSCP':  lambda s=subset: unified_cfa.weighted_score_by_performance(norm_tes, s, perf),
                    'WRCP':  lambda s=subset: unified_cfa.weighted_rank_by_performance(norm_tes, s, perf),
                }
                for m in methods:
                    is_rank = m in ('ARC', 'WRCDS', 'WRCP')
                    results.append({'combination': name, 'method': m,
                        'n_models': r,
                        'score': unified_cfa.evaluate(fn[m](), tes_gt, 'accuracy', is_rank)})
            combos_done += 1
            if time_limit > 0 and (time() - t0) >= time_limit:
                timed_out = True
                break
        if timed_out:
            break

    elapsed = time() - t0
    if timed_out:
        print('[CFA] Time limit reached after %.2fs — evaluated %d/%d combinations'
              % (elapsed, combos_done, total_combos))
    else:
        print('[CFA] All %d combinations evaluated in %.2fs' % (total_combos, elapsed))

    results_df = pd.DataFrame(results).sort_values('score', ascending=False)

    # Show results table
    table = Table(title='CFA Fusion Results (accuracy)', expand=True)
    table.add_column('Combination', justify='left')
    table.add_column('Method', justify='center')
    table.add_column('Models', justify='center')
    table.add_column('Score', justify='center')
    for _, row in results_df.head(15).iterrows():
        table.add_row(
            str(row['combination']), str(row['method']),
            str(row['n_models']), '%.4f' % row['score'])
    console.print(table)

    # Save full results
    cfa_path = os.path.join(run_dir, 'cfa_results.csv')
    results_df.to_csv(cfa_path, index=False)
    print('[CFA] Full results saved to: %s' % cfa_path)

    best = results_df.iloc[0]
    print('[CFA] Best: %s (%s) = %.4f' % (
        best['combination'], best['method'], best['score']))


# ---------------------------------------------------------------------------
# Model runner with guards
# ---------------------------------------------------------------------------

def _run_model(name, train_fn, summary, mem_limit_gb, time_limit_sec):
    """Run a model's train(), wrapped in MemoryGuard and TimeGuard."""
    print('\n===== %s =====' % name)
    killed = False
    kill_reason = ''
    try:
        with (MemoryGuard(limit_gb=mem_limit_gb) if mem_limit_gb > 0
              else _nullcontext()):
            with (TimeGuard(limit_sec=time_limit_sec) if time_limit_sec > 0
                  else _nullcontext()):
                train_fn(summary=summary)
    except MemoryError:
        killed = True
        kill_reason = 'exceeded %.1f GB RAM limit' % mem_limit_gb
    except TimeoutError:
        killed = True
        kill_reason = 'exceeded %.1f s time limit' % time_limit_sec
    if killed:
        print('\n[Guard] %s — %s — KILLED' % (name, kill_reason))
        if summary:
            summary.finish_model(name, {'acc': 0, 'mcc': 0},
                                 {'acc': 0, 'mcc': 0}, 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    desc = 'the lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--path', help='path of pv data', type=str,
                        default='./data/stocknet-dataset/price/ourpped')
    parser.add_argument('-l', '--seq', help='length of history', type=int,
                        default=5)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=32)
    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
                        help='alpha for l2 regularizer')
    parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                        help='beta for adverarial loss')
    parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
                        help='epsilon to control the scale of noise')
    parser.add_argument('-s', '--step', help='steps to make prediction',
                        type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                        default=1024)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    parser.add_argument('-r', '--learning_rate', help='learning rate',
                        type=float, default=1e-2)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-q', '--model_path', help='path to load model',
                        type=str, default='./data/saved_model/acl18_alstm/exp')
    parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model',
                        default='./data/tmp/model')
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, pred')
    parser.add_argument('-m', '--model', type=str, default='all',
                        help='which model to run: all, rf, gb, mlp, lr, sgdlog, sgdhinge, pa, ridge, perc, lda, qda, gnb, nc, dt, et, ada, quantum, classical, oom')

    parser.add_argument('-f', '--fix_init', type=int, default=0,
                        help='use fixed initialization')
    parser.add_argument('-a', '--att', type=int, default=1,
                        help='use attention model')
    parser.add_argument('-w', '--week', type=int, default=0,
                        help='use week day data')
    parser.add_argument('-v', '--adv', type=int, default=0,
                        help='adversarial training')
    parser.add_argument('-hi', '--hinge_lose', type=int, default=1,
                        help='use hinge lose')
    parser.add_argument('-rl', '--reload', type=int, default=0,
                        help='use pre-trained parameters')
    parser.add_argument('-qd', '--qlstm_depth', type=int, default=1,
                        help='VQC depth for quantum LSTM')
    parser.add_argument('-qh', '--qlstm_hidden', type=int, default=2,
                        help='hidden size for quantum LSTM')
    parser.add_argument('-qi', '--qlstm_input', type=int, default=2,
                        help='compressed input dim for quantum LSTM')
    parser.add_argument('-qe', '--qlstm_epoch', type=int, default=0,
                        help='quantum LSTM epochs (0 to skip)')
    parser.add_argument('-qt', '--qlstm_time', type=float, default=10.0,
                        help='time budget in seconds for quantum LSTM')
    parser.add_argument('-t', '--timeout', type=int, default=0,
                        help='kill script after this many seconds (0=no limit)')
    parser.add_argument('--test_oom', type=int, default=0,
                        help='run OOM test model (0=skip, 1=run)')
    parser.add_argument('--oom_gb', type=float, default=2.0,
                        help='GB the OOM test model tries to allocate')
    parser.add_argument('--mem_limit', type=float, default=1.0,
                        help='per-model RAM limit in GB (0=no limit)')
    parser.add_argument('--time_limit', type=float, default=1.0,
                        help='per-model time limit in seconds (0=no limit)')
    parser.add_argument('--sklearn', type=int, default=0,
                        help='run sklearn models: random forest, gradient boost, mlp (0=skip, 1=run)')
    parser.add_argument('--cfa_time', type=float, default=1.0,
                        help='time limit in seconds for CFA combination (0=unlimited)')
    parser.add_argument('--cfa_models', type=str, default=None,
                        help='comma-separated model names to include in CFA (default: all)')
    parser.add_argument('--run_dir', type=str, default=None,
                        help='path to a previous run directory (for -o cfa)')
    args = parser.parse_args()

    # -m / --model selects which model(s) to run
    _MODEL_MAP = {
        'rf': 'RANDOM FOREST', 'gb': 'GRADIENT BOOST', 'mlp': 'MLP CLASSIFIER',
        'lr': 'LOGISTIC REGRESSION',
        'quantum': 'QUANTUM LSTM', 'classical': 'CLASSICAL ALSTM', 'oom': 'TEST OOM',
    }
    _MODEL_MAP = {
        'rf': 'RANDOM FOREST', 'gb': 'GRADIENT BOOST', 'mlp': 'MLP CLASSIFIER',
        'lr': 'LOGISTIC REGRESSION',
        'sgdlog': 'SGD LOG', 'sgdhinge': 'SGD HINGE',
        'pa': 'PASSIVE AGGRESSIVE', 'ridge': 'RIDGE', 'perc': 'PERCEPTRON',
        'lda': 'LDA', 'gnb': 'GAUSSIAN NB',
        'nc': 'NEAREST CENTROID', 'dt': 'DECISION TREE',
        'et': 'EXTRA TREES', 'ada': 'ADABOOST',
        'quantum': 'QUANTUM LSTM', 'classical': 'CLASSICAL ALSTM', 'oom': 'TEST OOM',
            'lsvc': 'LINEAR SVC',
    'hgb': 'HIST GBDT',
    'bagdt': 'BAGGING DT2',
    'dummymf': 'DUMMY MOSTFREQ',
    'dummystr': 'DUMMY STRATIFIED',
    'knn3': 'KNN-3',
    'knn11d': 'KNN-11-DIST',
    'sgdmh': 'SGD MODHUBER',
    'gnb1e8': 'GAUSSIAN NB 1e-8',
    'gnb1e7': 'GAUSSIAN NB 1e-7',

     }

    if args.model != 'all':
        key = args.model.lower()
        if key not in _MODEL_MAP:
            print('ERROR: --model must be one of: all, %s' % ', '.join(_MODEL_MAP.keys()))
            exit(1)
        args.test_oom = 0
        args.sklearn = 0
        args.qlstm_epoch = 0
        args.epoch = 0
        
        if key in ('rf','gb','mlp','lr','sgdlog','sgdhinge','pa','ridge','perc','lda','gnb','nc','dt','et','ada', 'lsvc'     'hgb',     'bagdt',     'dummymf',     'dummystr',     'knn3',     'knn11d',     'sgdmh',     'gnb1e8',     'gnb1e7'):
           args.sklearn = 1


        elif key == 'quantum':
            args.qlstm_epoch = max(args.qlstm_epoch, 1) or 1
        elif key == 'classical':
            args.epoch = max(args.epoch, 1) or 150
        elif key == 'oom':
            args.test_oom = 1

    if args.timeout > 0:
        def _timeout_handler(signum, frame):
            print('\n\nTIMEOUT: script exceeded %d seconds, exiting.' % args.timeout)
            os._exit(1)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.timeout)

    print(args)

    # CFA standalone — no data loading needed
    if args.action == 'cfa':
        if not args.run_dir:
            import glob as globmod
            runs = sorted(globmod.glob('data/run_*'))
            if not runs:
                print('ERROR: no run directories found. Train first or use --run_dir.')
                exit(1)
            args.run_dir = runs[-1]
            print('[CFA] Using most recent run: %s' % args.run_dir)
        cfa_filter = None
        if args.cfa_models:
            cfa_filter = [m.strip() for m in args.cfa_models.split(',')]
        run_cfa(args.run_dir, time_limit=args.cfa_time, models_filter=cfa_filter)
        exit(0)

    parameters = {
        'seq': int(args.seq),
        'unit': int(args.unit),
        'alp': float(args.alpha_l2),
        'bet': float(args.beta_adv),
        'eps': float(args.epsilon_adv),
        'lr': float(args.learning_rate)
    }

    if 'stocknet' in args.path:
        tra_date = '2014-01-02'
        val_date = '2015-08-03'
        tes_date = '2015-10-01'
    elif 'kdd17' in args.path:
        tra_date = '2007-01-03'
        val_date = '2015-01-02'
        tes_date = '2016-01-04'
    else:
        print('unexpected path: %s' % args.path)
        exit(0)

    pure_LSTM = AWLSTM(
        data_path=args.path,
        model_path=args.model_path,
        model_save_path=args.model_save_path,
        parameters=parameters,
        steps=args.step,
        epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
        tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
        hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
        reload=args.reload
    )

    if args.action == 'train':
        summary = RunSummary()

        # Register all models that will run
        only_name = _MODEL_MAP.get(args.model, '') if args.model != 'all' else ''
        if args.test_oom:
            summary.add_model('TEST OOM', 1)
        
        if args.sklearn:
           for sk_name in [
               'RANDOM FOREST', 'GRADIENT BOOST', 'MLP CLASSIFIER', 'LOGISTIC REGRESSION',
               'SGD LOG', 'SGD HINGE', 'PASSIVE AGGRESSIVE', 'RIDGE', 'PERCEPTRON',
               'LDA', 'GAUSSIAN NB', 'NEAREST CENTROID',
               'DECISION TREE', 'EXTRA TREES', 'ADABOOST',
               'LINEAR SVC', 'HIST GBDT', 'BAGGING DT2',
'DUMMY MOSTFREQ', 'DUMMY STRATIFIED',
'KNN-3', 'KNN-11-DIST',
'SGD MODHUBER', 'GAUSSIAN NB 1e-8', 'GAUSSIAN NB 1e-7',

                 ]:
               if not only_name or sk_name == only_name:
                  summary.add_model(sk_name, 1)



        if args.qlstm_epoch > 0:
            summary.add_model('QUANTUM LSTM', args.qlstm_epoch)
        if args.epoch > 0:
            summary.add_model('CLASSICAL ALSTM', args.epoch)
        summary.print()

        # TEST OOM model (if enabled) — should get killed by MemoryGuard
        if args.test_oom:
            oom = OOMTestTrainer(
                tra_pv=pure_LSTM.tra_pv, tra_gt=pure_LSTM.tra_gt,
                val_pv=pure_LSTM.val_pv, val_gt=pure_LSTM.val_gt,
                tes_pv=pure_LSTM.tes_pv, tes_gt=pure_LSTM.tes_gt,
                epochs=1, target_gb=args.oom_gb,
            )
            _run_model('TEST OOM', oom.train, summary, args.mem_limit,
                       args.time_limit)
            del oom
            gc.collect()

        # Sklearn models
        if args.sklearn:
            hinge = (args.hinge_lose == 1)
            data_args = dict(
                tra_pv=pure_LSTM.tra_pv, tra_gt=pure_LSTM.tra_gt,
                val_pv=pure_LSTM.val_pv, val_gt=pure_LSTM.val_gt,
                tes_pv=pure_LSTM.tes_pv, tes_gt=pure_LSTM.tes_gt,
                hinge=hinge,
            )
            only_name = _MODEL_MAP.get(args.model, '') if args.model != 'all' else ''
            
            for name, cls in [
    ('RANDOM FOREST', RandomForestTrainer),
    ('GRADIENT BOOST', GradientBoostTrainer),
    ('MLP CLASSIFIER', MLPTrainer),
    ('LOGISTIC REGRESSION', LogisticRegressionTrainer),
    ('SGD LOG', SGDLogTrainer),
    ('SGD HINGE', SGDHingeTrainer),
    ('PASSIVE AGGRESSIVE', PassiveAggressiveTrainer),
    ('RIDGE', RidgeTrainer),
    ('PERCEPTRON', PerceptronTrainer),
    ('LDA', LDATrainer),
    
    ('GAUSSIAN NB', GaussianNBTrainer),
    ('NEAREST CENTROID', NearestCentroidTrainer),
    ('DECISION TREE', DecisionTreeTrainer),
    ('EXTRA TREES', ExtraTreesTrainer),
    ('ADABOOST', AdaBoostTrainer),
    ('LINEAR SVC', LinearSVCTrainer),
('HIST GBDT', HistGBDTTrainer),
('BAGGING DT2', BaggingDTTrainer),
('DUMMY MOSTFREQ', DummyMostFreqTrainer),
('DUMMY STRATIFIED', DummyStratifiedTrainer),
('KNN-3', KNN3Trainer),
('KNN-11-DIST', KNN11DistTrainer),
('SGD MODHUBER', SGDModHuberTrainer),
('GAUSSIAN NB 1e-8', GaussianNB1e8Trainer),
('GAUSSIAN NB 1e-7', GaussianNB1e7Trainer),

]:
             if only_name and name != only_name:
                continue
             trainer = cls(**data_args)
             _run_model(name, trainer.train, summary, args.mem_limit, args.time_limit)
             del trainer
             gc.collect()









        # Quantum LSTM (if epochs > 0)
        if args.qlstm_epoch > 0:
            qt = QuantumTrainer(
                tra_pv=pure_LSTM.tra_pv, tra_gt=pure_LSTM.tra_gt,
                val_pv=pure_LSTM.val_pv, val_gt=pure_LSTM.val_gt,
                tes_pv=pure_LSTM.tes_pv, tes_gt=pure_LSTM.tes_gt,
                hidden_size=args.qlstm_hidden,
                vqc_depth=args.qlstm_depth,
                qlstm_input=args.qlstm_input,
                epochs=args.qlstm_epoch,
                batch_size=args.batch_size,
                lr=args.learning_rate,
                hinge=(args.hinge_lose == 1),
                time_budget=args.qlstm_time,
            )
            _run_model('QUANTUM LSTM', qt.train, summary, args.mem_limit,
                       args.time_limit)
            del qt
            gc.collect()

        # Classical ALSTM
        if args.epoch > 0:
            _run_model('CLASSICAL ALSTM', pure_LSTM.train, summary,
                       args.mem_limit, args.time_limit)

        # CFA combination of all model predictions
        cfa_filter = None
        if args.cfa_models:
            cfa_filter = [m.strip() for m in args.cfa_models.split(',')]
        run_cfa(summary.run_dir, summary=summary, time_limit=args.cfa_time,
                models_filter=cfa_filter)

        print()
        summary.print()
        print('Run saved to: %s' % os.path.abspath(summary.run_dir))
    elif args.action == 'test':
        pure_LSTM.test()
    elif args.action == 'report':
        for i in range(5):
            pure_LSTM.train()
    elif args.action == 'pred':
        pure_LSTM.predict_record()
    elif args.action == 'adv':
        pure_LSTM.predict_adv()
    elif args.action == 'latent':
        pure_LSTM.get_latent_rep()
