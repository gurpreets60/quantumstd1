import argparse
import copy
from datetime import datetime, timedelta
import numpy as np
import os
import signal
import pathlib
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# Auto-detect pip-installed NVIDIA CUDA libs for GPU support
_nvidia_dir = pathlib.Path(__file__).resolve().parent / '.venv' / 'lib'
for _lib in _nvidia_dir.rglob('nvidia/*/lib'):
    if _lib.is_dir():
        os.environ['LD_LIBRARY_PATH'] = str(_lib) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
import pennylane as qml
import psutil
import random
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error
from sklearn.utils import shuffle
import subprocess
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from time import time
import torch
import torch.nn as nn


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
# Quantum LSTM (PennyLane + PyTorch)
# ---------------------------------------------------------------------------

def _qlstm_h_layer(nqubits):
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)

def _qlstm_ry_layer(w):
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def _qlstm_entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

def _qlstm_circuit(x, q_weights, n_class):
    n_dep = q_weights.shape[0]
    n_qub = q_weights.shape[1]
    _qlstm_h_layer(n_qub)
    _qlstm_ry_layer(x)
    for k in range(n_dep):
        _qlstm_entangling_layer(n_qub)
        _qlstm_ry_layer(q_weights[k])
    return [qml.expval(qml.PauliZ(p)) for p in range(n_class)]


class VQC(nn.Module):
    def __init__(self, vqc_depth, n_qubits, n_class):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.circuit = qml.QNode(_qlstm_circuit, self.dev, interface='torch')
        self.n_class = n_class

    def forward(self, X):
        return torch.stack([
            torch.stack(self.circuit(x, self.weights, self.n_class)) for x in X
        ])


class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vqc_depth):
        super().__init__()
        self.hidden_size = hidden_size
        n_qubits = input_size + hidden_size
        self.input_gate = VQC(vqc_depth, n_qubits, hidden_size)
        self.forget_gate = VQC(vqc_depth, n_qubits, hidden_size)
        self.cell_gate = VQC(vqc_depth, n_qubits, hidden_size)
        self.output_gate = VQC(vqc_depth, n_qubits, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)
        i_t = torch.sigmoid(self.input_gate(combined))
        f_t = torch.sigmoid(self.forget_gate(combined))
        g_t = torch.tanh(self.cell_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        out = self.output_fc(h_t)
        return out, h_t, c_t


class QLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vqc_depth, qlstm_input=2):
        super().__init__()
        self.hidden_size = hidden_size
        # Compress high-dim features down to qlstm_input before quantum circuit
        self.compress = nn.Linear(input_size, qlstm_input)
        self.cell = QLSTMCell(qlstm_input, hidden_size, output_size, vqc_depth)

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        x = self.compress(x)
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h_t, c_t = hidden
        outputs = []
        for t in range(seq_len):
            out, h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(out.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h_t, c_t)


class QuantumTrainer:
    """Train a QLSTM on the same stock data used by the classical AWLSTM."""

    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hidden_size=2, vqc_depth=1, qlstm_input=2, epochs=10,
                 batch_size=256, lr=0.01, hinge=True, time_budget=10.0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.hinge = hinge
        self.time_budget = time_budget

        input_size = tra_pv.shape[2]
        self.model = QLSTMModel(input_size, hidden_size, 1, vqc_depth,
                                qlstm_input=qlstm_input).double()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Convert data
        tra_x = torch.from_numpy(tra_pv).double()
        tra_y = torch.from_numpy(tra_gt).double()
        self.val_x = torch.from_numpy(val_pv).double()
        self.val_y = torch.from_numpy(val_gt).double()
        self.tes_x = torch.from_numpy(tes_pv).double()
        self.tes_y = torch.from_numpy(tes_gt).double()

        # Auto-calibrate: time a pilot of 2 samples, then cap all sets
        pilot_n = min(2, tra_x.size(0))
        self.model.eval()
        with torch.no_grad():
            t0 = time()
            self.model(tra_x[:pilot_n])
            pilot_sec = time() - t0
        sec_per_sample = pilot_sec / pilot_n

        # Budget per epoch: split between train (50%), val+test (50%)
        per_epoch_budget = self.time_budget / max(self.epochs, 1)
        eval_budget = per_epoch_budget * 0.5
        train_budget = per_epoch_budget * 0.5

        # Cap training samples (3x cost: fwd + bwd + optim)
        max_train = max(int(train_budget / (sec_per_sample * 3)), pilot_n)
        if max_train < tra_x.size(0):
            tra_x = tra_x[torch.randperm(tra_x.size(0))[:max_train]]
            tra_y = tra_y[:max_train]
        self.tra_x = tra_x
        self.tra_y = tra_y

        # Cap val/test samples too (1x cost: fwd only)
        max_eval = max(int(eval_budget / (sec_per_sample * 2)), pilot_n)
        if max_eval < self.val_x.size(0):
            perm = torch.randperm(self.val_x.size(0))[:max_eval]
            self.val_x = self.val_x[perm]
            self.val_y = self.val_y[perm]
        if max_eval < self.tes_x.size(0):
            perm = torch.randperm(self.tes_x.size(0))[:max_eval]
            self.tes_x = self.tes_x[perm]
            self.tes_y = self.tes_y[perm]

        n_qubits = qlstm_input + hidden_size
        print(f'[QUANTUM LSTM] {n_qubits} qubits, depth={vqc_depth}, '
              f'compress {input_size}->{qlstm_input}, '
              f'~{sec_per_sample:.4f}s/sample, '
              f'using {self.tra_x.size(0)}/{tra_pv.shape[0]} train samples')

    def _eval_perf(self, pred_np, gt_np):
        if self.hinge:
            binary_pred = np.where(pred_np > 0.5, 1.0, 0.0)
        else:
            binary_pred = np.round(pred_np)
        acc = accuracy_score(gt_np, binary_pred)
        mcc = matthews_corrcoef(gt_np, binary_pred)
        return {'acc': acc, 'mcc': mcc}

    def train(self, summary=None):
        if summary:
            summary.start_model('QUANTUM LSTM')
        monitor = SystemMonitor(model_name='QUANTUM LSTM', total_epochs=self.epochs, summary=summary)
        best_valid_perf = {'acc': 0, 'mcc': -2}
        best_test_perf = {'acc': 0, 'mcc': -2}
        train_t0 = time()

        monitor.start()
        for epoch in range(self.epochs):
            t1 = time()
            monitor.update(f'Training batches (epoch {epoch}/{self.epochs})')

            # Training
            self.model.train()
            perm = torch.randperm(self.tra_x.size(0))
            tra_x_shuf = self.tra_x[perm]
            tra_y_shuf = self.tra_y[perm]
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, tra_x_shuf.size(0), self.batch_size):
                xb = tra_x_shuf[start:start + self.batch_size]
                yb = tra_y_shuf[start:start + self.batch_size]
                self.optimizer.zero_grad()
                out, _ = self.model(xb)
                loss = self.loss_fn(out[:, -1, :], yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            monitor.log('----->>>>> Training loss: %.6f' % avg_loss)

            # Validation
            monitor.update('Evaluating validation set')
            self.model.eval()
            with torch.no_grad():
                val_out, _ = self.model(self.val_x)
                val_pred = val_out[:, -1, :].numpy()
            val_perf = self._eval_perf(val_pred, self.val_y.numpy())
            monitor.log('\tVal per: %s' % val_perf)

            # Test
            monitor.update('Evaluating test set')
            with torch.no_grad():
                tes_out, _ = self.model(self.tes_x)
                tes_pred = tes_out[:, -1, :].numpy()
            tes_perf = self._eval_perf(tes_pred, self.tes_y.numpy())
            monitor.log('\tTest per: %s' % tes_perf)

            if val_perf['acc'] > best_valid_perf['acc']:
                best_valid_perf = copy.copy(val_perf)
                best_test_perf = copy.copy(tes_perf)

            t4 = time()
            monitor.epoch_done(epoch, t4 - t1)
            monitor.log('epoch: %d time: %.4f' % (epoch, t4 - t1))

        monitor.stop()
        print('\n[QUANTUM LSTM] Best Valid performance:', best_valid_perf)
        print('\t[QUANTUM LSTM] Best Test performance:', best_test_perf)
        if summary:
            summary.finish_model('QUANTUM LSTM', best_valid_perf, best_test_perf, time() - train_t0)
        return best_valid_perf, best_test_perf


def load_cla_data(data_path, tra_date, val_date, tes_date, seq=2,
                  date_format='%Y-%m-%d'):
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    print(len(fnames), ' tickers selected')

    data_EOD = []
    for index, fname in enumerate(fnames):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, fname), dtype=float, delimiter=',',
            skip_header=False
        )
        data_EOD.append(single_EOD)
    fea_dim = data_EOD[0].shape[1] - 2

    trading_dates = np.genfromtxt(
        os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    print(len(trading_dates), 'trading dates:')

    dates_index = {}
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]
    print(tra_ind, val_ind, tes_ind)

    # count training, validation, and testing instances
    tra_num = 0
    val_num = 0
    tes_num = 0
    for date_ind in range(tra_ind, val_ind):
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tra_num += 1
    print(tra_num, ' training instances')

    for date_ind in range(val_ind, tes_ind):
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    val_num += 1
    print(val_num, ' validation instances')

    for date_ind in range(tes_ind, len(trading_dates)):
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tes_num += 1
    print(tes_num, ' testing instances')

    # generate training, validation, and testing instances
    tra_pv = np.zeros([tra_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tra_ind, val_ind):
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                    data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tra_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, : -2]
                tra_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tra_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1

    val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(val_ind, tes_ind):
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                val_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                val_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                val_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1

    tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tes_ind, len(trading_dates)):
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tes_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                tes_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tes_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1
    return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt


def evaluate(prediction, ground_truth, hinge=False, reg=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    if reg:
        performance['mse'] = mean_squared_error(np.squeeze(ground_truth), np.squeeze(prediction))
        return performance

    if hinge:
        pred = (np.sign(prediction) + 1) / 2
        for ind, p in enumerate(pred):
            v = p[0]
            if abs(p[0] - 0.5) < 1e-8 or np.isnan(p[0]):
                pred[ind][0] = 0
    else:
        pred = np.round(prediction)
    try:
        performance['acc'] = accuracy_score(ground_truth, pred)
    except Exception:
        np.savetxt('prediction', pred, delimiter=',')
        exit(0)
    performance['mcc'] = matthews_corrcoef(ground_truth, pred)
    return performance


def compare(current_performance, origin_performance):
    is_better = {}
    for metric_name in origin_performance.keys():
        if metric_name == 'mse':
            if current_performance[metric_name] < \
                    origin_performance[metric_name]:
                is_better[metric_name] = True
            else:
                is_better[metric_name] = False
        else:
            if current_performance[metric_name] > \
                    origin_performance[metric_name]:
                is_better[metric_name] = True
            else:
                is_better[metric_name] = False
    return is_better

class AWLSTM:
    def __init__(self, data_path, model_path, model_save_path, parameters, steps=1, epochs=50,
                 batch_size=256, gpu=False, tra_date='2014-01-02',
                 val_date='2015-08-03', tes_date='2015-10-01', att=0, hinge=0,
                 fix_init=0, adv=0, reload=0):
        self.data_path = data_path
        self.model_path = model_path
        self.model_save_path = model_save_path
        # model parameters
        self.paras = copy.copy(parameters)
        # training parameters
        self.steps = steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.gpu = gpu

        if att == 1:
            self.att = True
        else:
            self.att = False
        if hinge == 1:
            self.hinge = True
        else:
            self.hinge = False
        if fix_init == 1:
            self.fix_init = True
        else:
            self.fix_init = False
        if adv == 1:
            self.adv_train = True
        else:
            self.adv_train = False
        if reload == 1:
            self.reload = True
        else:
            self.reload = False

        # load data
        self.tra_date = tra_date
        self.val_date = val_date
        self.tes_date = tes_date
        self.tra_pv, self.tra_wd, self.tra_gt, \
        self.val_pv, self.val_wd, self.val_gt, \
        self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data(
            self.data_path,
            tra_date, val_date, tes_date, seq=self.paras['seq']
        )
        self.fea_dim = self.tra_pv.shape[2]

    def get_batch(self, sta_ind=None):
        if sta_ind is None:
            sta_ind = random.randrange(0, self.tra_pv.shape[0])
        if sta_ind + self.batch_size < self.tra_pv.shape[0]:
            end_ind = sta_ind + self.batch_size
        else:
            sta_ind = self.tra_pv.shape[0] - self.batch_size
            end_ind = self.tra_pv.shape[0]
        return self.tra_pv[sta_ind:end_ind, :, :], \
               self.tra_wd[sta_ind:end_ind, :, :], \
               self.tra_gt[sta_ind:end_ind, :]

    def adv_part(self, adv_inputs):
        print('adversial part')
        if self.att:
            with tf.variable_scope('pre_fc'):
                self.fc_W = tf.get_variable(
                    'weights', dtype=tf.float32,
                    shape=[self.paras['unit'] * 2, 1],
                    initializer=tf.glorot_uniform_initializer()
                )
                self.fc_b = tf.get_variable(
                    'biases', dtype=tf.float32,
                    shape=[1, ],
                    initializer=tf.zeros_initializer()
                )
                if self.hinge:
                    pred = tf.nn.bias_add(
                        tf.matmul(adv_inputs, self.fc_W), self.fc_b
                    )
                else:
                    pred = tf.nn.sigmoid(
                        tf.nn.bias_add(tf.matmul(self.fea_con, self.fc_W),
                                       self.fc_b)
                    )
        else:
            # One hidden layer
            if self.hinge:
                pred = tf.layers.dense(
                    adv_inputs, units=1, activation=None,
                    name='pre_fc',
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
            else:
                pred = tf.layers.dense(
                    adv_inputs, units=1, activation=tf.nn.sigmoid,
                    name='pre_fc',
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
        return pred

    def construct_graph(self):
        print('is pred_lstm')
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()
            if self.fix_init:
                tf.set_random_seed(123456)

            self.gt_var = tf.placeholder(tf.float32, [None, 1])
            self.pv_var = tf.placeholder(
                tf.float32, [None, self.paras['seq'], self.fea_dim]
            )
            self.wd_var = tf.placeholder(
                tf.float32, [None, self.paras['seq'], 5]
            )

            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                self.paras['unit']
            )

            # self.outputs, _ = tf.nn.dynamic_rnn(
            #     # self.outputs, _ = tf.nn.static_rnn(
            #     self.lstm_cell, self.pv_var, dtype=tf.float32
            #     # , initial_state=ini_sta
            # )

            self.in_lat = tf.layers.dense(
                self.pv_var, units=self.fea_dim,
                activation=tf.nn.tanh, name='in_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            self.outputs, _ = tf.nn.dynamic_rnn(
                # self.outputs, _ = tf.nn.static_rnn(
                self.lstm_cell, self.in_lat, dtype=tf.float32
                # , initial_state=ini_sta
            )

            self.loss = 0
            self.adv_loss = 0
            self.l2_norm = 0
            if self.att:
                with tf.variable_scope('lstm_att') as scope:
                    self.av_W = tf.get_variable(
                        name='att_W', dtype=tf.float32,
                        shape=[self.paras['unit'], self.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )
                    self.av_b = tf.get_variable(
                        name='att_h', dtype=tf.float32,
                        shape=[self.paras['unit']],
                        initializer=tf.zeros_initializer()
                    )
                    self.av_u = tf.get_variable(
                        name='att_u', dtype=tf.float32,
                        shape=[self.paras['unit']],
                        initializer=tf.glorot_uniform_initializer()
                    )

                    self.a_laten = tf.tanh(
                        tf.tensordot(self.outputs, self.av_W,
                                     axes=1) + self.av_b)
                    self.a_scores = tf.tensordot(self.a_laten, self.av_u,
                                                 axes=1,
                                                 name='scores')
                    self.a_alphas = tf.nn.softmax(self.a_scores, name='alphas')

                    self.a_con = tf.reduce_sum(
                        self.outputs * tf.expand_dims(self.a_alphas, -1), 1)
                    self.fea_con = tf.concat(
                        [self.outputs[:, -1, :], self.a_con],
                        axis=1)
                    print('adversarial scope')
                    # training loss
                    self.pred = self.adv_part(self.fea_con)
                    if self.hinge:
                        self.loss = tf.losses.hinge_loss(self.gt_var, self.pred)
                    else:
                        self.loss = tf.losses.log_loss(self.gt_var, self.pred)

                    self.adv_loss = self.loss * 0

                    # adversarial loss
                    if self.adv_train:
                        print('gradient noise')
                        self.delta_adv = tf.gradients(self.loss, [self.fea_con])[0]
                        tf.stop_gradient(self.delta_adv)
                        self.delta_adv = tf.nn.l2_normalize(self.delta_adv, axis=1)
                        self.adv_pv_var = self.fea_con + \
                                          self.paras['eps'] * self.delta_adv

                        scope.reuse_variables()
                        self.adv_pred = self.adv_part(self.adv_pv_var)
                        if self.hinge:
                            self.adv_loss = tf.losses.hinge_loss(self.gt_var, self.adv_pred)
                        else:
                            self.adv_loss = tf.losses.log_loss(self.gt_var, self.adv_pred)
            else:
                with tf.variable_scope('lstm_att') as scope:
                    print('adversarial scope')
                    # training loss
                    self.pred = self.adv_part(self.outputs[:, -1, :])
                    if self.hinge:
                        self.loss = tf.losses.hinge_loss(self.gt_var, self.pred)
                    else:
                        self.loss = tf.losses.log_loss(self.gt_var, self.pred)

                    self.adv_loss = self.loss * 0

                    # adversarial loss
                    if self.adv_train:
                        print('gradient noise')
                        self.delta_adv = tf.gradients(self.loss, [self.outputs[:, -1, :]])[0]
                        tf.stop_gradient(self.delta_adv)
                        self.delta_adv = tf.nn.l2_normalize(self.delta_adv,
                                                            axis=1)
                        self.adv_pv_var = self.outputs[:, -1, :] + \
                                          self.paras['eps'] * self.delta_adv

                        scope.reuse_variables()
                        self.adv_pred = self.adv_part(self.adv_pv_var)
                        if self.hinge:
                            self.adv_loss = tf.losses.hinge_loss(self.gt_var,
                                                                 self.adv_pred)
                        else:
                            self.adv_loss = tf.losses.log_loss(self.gt_var,
                                                               self.adv_pred)

            # regularizer
            self.tra_vars = tf.trainable_variables('lstm_att/pre_fc')
            for var in self.tra_vars:
                self.l2_norm += tf.nn.l2_loss(var)

            self.obj_func = self.loss + \
                            self.paras['alp'] * self.l2_norm + \
                            self.paras['bet'] * self.adv_loss

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.paras['lr']
            ).minimize(self.obj_func)

    def get_latent_rep(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1

        tr_lat_rep = np.zeros([bat_count * self.batch_size, self.paras['unit'] * 2],
                              dtype=np.float32)
        tr_gt = np.zeros([bat_count * self.batch_size, 1], dtype=np.float32)
        for j in range(bat_count):
            pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
            feed_dict = {
                self.pv_var: pv_b,
                self.wd_var: wd_b,
                self.gt_var: gt_b
            }
            lat_rep, cur_obj, cur_loss, cur_l2, cur_al = sess.run(
                (self.fea_con, self.obj_func, self.loss, self.l2_norm,
                 self.adv_loss),
                feed_dict
            )
            print(lat_rep.shape)
            tr_lat_rep[j * self.batch_size: (j + 1) * self.batch_size, :] = lat_rep
            tr_gt[j * self.batch_size: (j + 1) * self.batch_size,:] = gt_b

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_lat_rep, val_pre = sess.run(
            (self.loss, self.fea_con, self.pred), feed_dict
        )
        cur_val_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per:', cur_val_perf)

        sess.close()
        tf.reset_default_graph()
        np.savetxt(self.model_save_path + '_val_lat_rep.csv', val_lat_rep)
        np.savetxt(self.model_save_path + '_tr_lat_rep.csv', tr_lat_rep)
        np.savetxt(self.model_save_path + '_val_gt.csv', self.val_gt)
        np.savetxt(self.model_save_path + '_tr_gt.csv', tr_gt)

    def predict_adv(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1
        tra_perf = None
        adv_perf = None
        for j in range(bat_count):
            pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
            feed_dict = {
                self.pv_var: pv_b,
                self.wd_var: wd_b,
                self.gt_var: gt_b
            }
            cur_pre, cur_adv_pre, cur_obj, cur_loss, cur_l2, cur_al = sess.run(
                (self.pred, self.adv_pred, self.obj_func, self.loss, self.l2_norm,
                 self.adv_loss),
                feed_dict
            )
            cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
            cur_adv_perf = evaluate(cur_adv_pre, gt_b, self.hinge)
            if tra_perf is None:
                tra_perf = copy.copy(cur_tra_perf)
            else:
                for metric in tra_perf.keys():
                    tra_perf[metric] = tra_perf[metric] + cur_tra_perf[metric]
            if adv_perf is None:
                adv_perf = copy.copy(cur_adv_perf)
            else:
                for metric in adv_perf.keys():
                    adv_perf[metric] = adv_perf[metric] + cur_adv_perf[metric]
        for metric in tra_perf.keys():
            tra_perf[metric] = tra_perf[metric] / bat_count
            adv_perf[metric] = adv_perf[metric] / bat_count

        print('Clean samples performance:', tra_perf)
        print('Adversarial samples performance:', adv_perf)

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_pre, val_adv_pre = sess.run(
            (self.loss, self.pred, self.adv_pred), feed_dict
        )
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per clean:', cur_valid_perf)
        adv_valid_perf = evaluate(val_adv_pre, self.val_gt, self.hinge)
        print('\tVal per adversarial:', adv_valid_perf)

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt
        }
        test_loss, tes_pre, tes_adv_pre = sess.run(
            (self.loss, self.pred, self.adv_pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        print('\tTest per clean:', cur_test_perf)
        adv_test_perf = evaluate(tes_adv_pre, self.tes_gt, self.hinge)
        print('\tTest per adversarial:', adv_test_perf)

        sess.close()
        tf.reset_default_graph()

    def predict_record(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)
        np.savetxt(self.model_save_path + '_val_prediction.csv', val_pre)

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt
        }
        test_loss, tes_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)
        np.savetxt(self.model_save_path + '_tes_prediction.csv', tes_pre)
        sess.close()
        tf.reset_default_graph()

    def test(self):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        # test on validation set
        feed_dict = {
            self.pv_var: self.val_pv,
            self.wd_var: self.val_wd,
            self.gt_var: self.val_gt
        }
        val_loss, val_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
        print('\tVal per:', cur_valid_perf, '\tVal loss:', val_loss)

        # test on testing set
        feed_dict = {
            self.pv_var: self.tes_pv,
            self.wd_var: self.tes_wd,
            self.gt_var: self.tes_gt
        }
        test_loss, tes_pre = sess.run(
            (self.loss, self.pred), feed_dict
        )
        cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
        print('\tTest per:', cur_test_perf, '\tTest loss:', test_loss)
        sess.close()
        tf.reset_default_graph()

    def train(self, tune_para=False, summary=None):
        self.construct_graph()

        sess = tf.Session()
        saver = tf.train.Saver()
        if self.reload:
            saver.restore(sess, self.model_path)
            print('model restored')
        else:
            sess.run(tf.global_variables_initializer())

        best_valid_pred = np.zeros(self.val_gt.shape, dtype=float)
        best_test_pred = np.zeros(self.tes_gt.shape, dtype=float)

        best_valid_perf = {
            'acc': 0, 'mcc': -2
        }
        best_test_perf = {
            'acc': 0, 'mcc': -2
        }

        monitor = SystemMonitor(model_name='CLASSICAL ALSTM', total_epochs=self.epochs, summary=summary)
        if summary:
            summary.start_model('CLASSICAL ALSTM')
        train_t0 = time()
        bat_count = self.tra_pv.shape[0] // self.batch_size
        if not (self.tra_pv.shape[0] % self.batch_size == 0):
            bat_count += 1
        monitor.start()
        for i in range(self.epochs):
            t1 = time()
            monitor.update(f'Training batches (epoch {i}/{self.epochs})')
            tra_loss = 0.0
            tra_obj = 0.0
            l2 = 0.0
            tra_adv = 0.0
            for j in range(bat_count):
                pv_b, wd_b, gt_b = self.get_batch(j * self.batch_size)
                feed_dict = {
                    self.pv_var: pv_b,
                    self.wd_var: wd_b,
                    self.gt_var: gt_b
                }
                cur_pre, cur_obj, cur_loss, cur_l2, cur_al, batch_out = sess.run(
                    (self.pred, self.obj_func, self.loss, self.l2_norm, self.adv_loss,
                     self.optimizer),
                    feed_dict
                )

                tra_loss += cur_loss
                tra_obj += cur_obj
                l2 += cur_l2
                tra_adv += cur_al
            monitor.log('----->>>>> Training: %s %s %s %s' % (
                tra_obj / bat_count, tra_loss / bat_count,
                l2 / bat_count, tra_adv / bat_count))

            if not tune_para:
                monitor.update('Evaluating training accuracy')
                tra_loss = 0.0
                tra_obj = 0.0
                l2 = 0.0
                tra_acc = 0.0
                for j in range(bat_count):
                    pv_b, wd_b, gt_b = self.get_batch(
                        j * self.batch_size)
                    feed_dict = {
                        self.pv_var: pv_b,
                        self.wd_var: wd_b,
                        self.gt_var: gt_b
                    }
                    cur_obj, cur_loss, cur_l2, cur_pre = sess.run(
                        (self.obj_func, self.loss, self.l2_norm, self.pred),
                        feed_dict
                    )
                    cur_tra_perf = evaluate(cur_pre, gt_b, self.hinge)
                    tra_loss += cur_loss
                    l2 += cur_l2
                    tra_obj += cur_obj
                    tra_acc += cur_tra_perf['acc']
                monitor.log('Training: %s %s %s \tTrain per: %s' % (
                    tra_obj / bat_count, tra_loss / bat_count,
                    l2 / bat_count, tra_acc / bat_count))

            # test on validation set
            monitor.update('Evaluating validation set')
            feed_dict = {
                self.pv_var: self.val_pv,
                self.wd_var: self.val_wd,
                self.gt_var: self.val_gt
            }
            val_loss, val_pre = sess.run(
                (self.loss, self.pred), feed_dict
            )
            cur_valid_perf = evaluate(val_pre, self.val_gt, self.hinge)
            monitor.log('\tVal per: %s \tVal loss: %s' % (
                cur_valid_perf, val_loss))

            # test on testing set
            monitor.update('Evaluating test set')
            feed_dict = {
                self.pv_var: self.tes_pv,
                self.wd_var: self.tes_wd,
                self.gt_var: self.tes_gt
            }
            test_loss, tes_pre = sess.run(
                (self.loss, self.pred), feed_dict
            )
            cur_test_perf = evaluate(tes_pre, self.tes_gt, self.hinge)
            monitor.log('\tTest per: %s \tTest loss: %s' % (
                cur_test_perf, test_loss))

            if cur_valid_perf['acc'] > best_valid_perf['acc']:
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_pred = copy.copy(val_pre)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_pred = copy.copy(tes_pre)
                if not tune_para:
                    saver.save(sess, self.model_save_path)
            self.tra_pv, self.tra_wd, self.tra_gt = shuffle(
                self.tra_pv, self.tra_wd, self.tra_gt, random_state=0
            )
            t4 = time()
            monitor.epoch_done(i, t4 - t1)
            monitor.log('epoch: %d time: %.4f' % (i, t4 - t1))
        monitor.stop()
        print('\n[CLASSICAL ALSTM] Best Valid performance:', best_valid_perf)
        print('\t[CLASSICAL ALSTM] Best Test performance:', best_test_perf)
        if summary:
            summary.finish_model('CLASSICAL ALSTM', best_valid_perf, best_test_perf, time() - train_t0)
        sess.close()
        tf.reset_default_graph()
        if tune_para:
            return best_valid_perf, best_test_perf
        return best_valid_pred, best_test_pred

    def update_model(self, parameters):
        data_update = False
        if not parameters['seq'] == self.paras['seq']:
            data_update = True
        for name, value in parameters.items():
            self.paras[name] = value
        if data_update:
            self.tra_pv, self.tra_wd, self.tra_gt, \
            self.val_pv, self.val_wd, self.val_gt, \
            self.tes_pv, self.tes_wd, self.tes_gt = load_cla_data(
                self.data_path,
                self.tra_date, self.val_date, self.tes_date, seq=self.paras['seq']
            )
        return True

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
    parser.add_argument('-m', '--model', type=str, default='pure_lstm',
                        help='pure_lstm, di_lstm, att_lstm, week_lstm, aw_lstm')
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
    args = parser.parse_args()

    if args.timeout > 0:
        def _timeout_handler(signum, frame):
            print('\n\nTIMEOUT: script exceeded %d seconds, exiting.' % args.timeout)
            os._exit(1)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.timeout)

    print(args)

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
        if args.qlstm_epoch > 0:
            summary.add_model('QUANTUM LSTM', args.qlstm_epoch)
        if args.epoch > 0:
            summary.add_model('CLASSICAL ALSTM', args.epoch)
        summary.print()

        # Quantum LSTM first (if epochs > 0)
        if args.qlstm_epoch > 0:
            print('\n===== QUANTUM LSTM =====')
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
            qt.train(summary=summary)

        # Classical ALSTM second
        if args.epoch > 0:
            print('\n===== CLASSICAL ALSTM =====')
            pure_LSTM.train(summary=summary)

        print()
        summary.print()
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