import csv
from datetime import datetime, timedelta
import os
import psutil
import subprocess
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
        self._csv_path = os.path.join(
            'data', 'results_%s.csv' % datetime.now().strftime('%Y%m%d_%H%M%S')
        )

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
