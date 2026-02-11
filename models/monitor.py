import csv
import ctypes
from datetime import datetime, timedelta
import gc
import io
import numpy as np
import os
import psutil
import subprocess
import threading
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from time import time


class SystemMonitor:
    def __init__(self, model_name='', total_epochs=0, summary=None):
        psutil.cpu_percent(interval=None)
        self._has_gpu = self._check_gpu()
        self._console = Console()
        self._live = Live(console=self._console, refresh_per_second=4)
        self._phase = ''
        self._model_name = model_name
        self._total_epochs = total_epochs
        self._current_epoch = 0
        self._epoch_times = []
        self._train_start = None
        self._summary = summary

    def _check_gpu(self):
        try:
            subprocess.run(
                ['nvidia-smi'], capture_output=True, check=True
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def _gpu_stats(self):
        if not self._has_gpu:
            return 'N/A', '', ''
        try:
            out = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            ).stdout.strip()
            util, mem_used, mem_total = [x.strip() for x in out.split(',')]
            return f'{util}%', f'{mem_used} MB', f'{mem_total} MB'
        except Exception:
            return 'N/A', '', ''

    def _format_eta(self):
        if not self._epoch_times or self._current_epoch >= self._total_epochs:
            return 'N/A'
        avg_time = sum(self._epoch_times) / len(self._epoch_times)
        remaining = self._total_epochs - self._current_epoch
        eta_secs = avg_time * remaining
        eta = str(timedelta(seconds=int(eta_secs)))
        return eta

    def _build_table(self):
        mem = psutil.virtual_memory()
        ram_used = mem.used / (1024 ** 3)
        ram_total = mem.total / (1024 ** 3)
        cpu = psutil.cpu_percent(interval=None)
        gpu_util, gpu_used, gpu_total = self._gpu_stats()
        eta = self._format_eta()

        title = f'{self._model_name} | {self._phase}'
        table = Table(title=title, expand=True)
        table.add_column('RAM', justify='center')
        table.add_column('CPU', justify='center')
        table.add_column('GPU', justify='center')
        table.add_column('VRAM', justify='center')
        table.add_column('ETA', justify='center')

        vram = f'{gpu_used}/{gpu_total}' if gpu_used else 'N/A'
        table.add_row(
            f'{ram_used:.1f}/{ram_total:.1f} GB ({mem.percent}%)',
            f'{cpu}%',
            gpu_util,
            vram,
            eta,
        )
        if self._summary:
            return Group(self._summary.build_table(), table)
        return table

    def start(self):
        self._train_start = time()
        self._live.start()

    def stop(self):
        self._live.stop()

    def epoch_done(self, epoch, epoch_time):
        self._current_epoch = epoch + 1
        self._epoch_times.append(epoch_time)

    def update(self, phase):
        self._phase = phase
        self._live.update(self._build_table())

    def log(self, message):
        self._live.console.print(message, highlight=False)


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


class MemoryGuard:
    """Enforces a per-model RAM limit. Kills the model if it exceeds it.

    Usage::

        with MemoryGuard(limit_gb=1.0) as guard:
            model.train(summary=summary)
        # If model exceeded 1 GB *above baseline*, a MemoryError is raised.
    """

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
        # Don't suppress exceptions — let MemoryError propagate
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
                # Inject MemoryError into the main thread
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(main_tid),
                    ctypes.py_object(MemoryError),
                )
                break
            self._stop.wait(self.check_interval)

    @property
    def exceeded(self):
        return self._exceeded


class TimeGuard:
    """Enforces a per-model time limit. Kills the model if it runs too long.

    Usage::

        with TimeGuard(limit_sec=60):
            model.train(summary=summary)
        # If model ran longer than 60 seconds, a TimeoutError is raised.
    """

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
