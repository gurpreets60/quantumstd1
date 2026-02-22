"""Batch-friendly QLSTM trainer wired to the project training pipeline."""

import copy
from time import time

import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch
import torch.nn as nn

from .monitor import SystemMonitor


# ---------------------------------------------------------------------------
# Quantum circuit blocks
# ---------------------------------------------------------------------------

def _h_layer(n_qubits):
    """Apply a Hadamard layer to all qubits."""
    for wire in range(n_qubits):
        qml.Hadamard(wires=wire)


def _ry_layer(values):
    """Encode a vector with RY rotations."""
    for wire, value in enumerate(values):
        qml.RY(value, wires=wire)


def _entangle_layer(n_qubits):
    """Use a light nearest-neighbor entangling pattern."""
    for wire in range(0, n_qubits - 1, 2):
        qml.CNOT(wires=[wire, wire + 1])
    for wire in range(1, n_qubits - 1, 2):
        qml.CNOT(wires=[wire, wire + 1])


def _qlstm_circuit(x, q_weights, n_outputs):
    """Small variational circuit used by each LSTM gate."""
    n_depth = q_weights.shape[0]
    n_qubits = q_weights.shape[1]
    _h_layer(n_qubits)
    _ry_layer(x)
    for depth in range(n_depth):
        _entangle_layer(n_qubits)
        _ry_layer(q_weights[depth])
    return [qml.expval(qml.PauliZ(wire)) for wire in range(n_outputs)]


# ---------------------------------------------------------------------------
# QLSTM model
# ---------------------------------------------------------------------------

class _BatchVQC(nn.Module):
    """PyTorch wrapper for a single variational quantum circuit."""

    def __init__(self, vqc_depth, n_qubits, n_outputs):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))
        self.n_outputs = n_outputs
        dev = qml.device('default.qubit', wires=n_qubits)
        self.qnode = qml.QNode(_qlstm_circuit, dev, interface='torch')

    def forward(self, x_batch):
        # Keep this explicit per-sample loop for PennyLane compatibility.
        rows = [
            torch.stack(self.qnode(row, self.weights, self.n_outputs))
            for row in x_batch
        ]
        return torch.stack(rows, dim=0)


class _BatchQLSTMCell(nn.Module):
    """One QLSTM cell with quantum gates and a linear readout."""

    def __init__(self, input_size, hidden_size, output_size, vqc_depth):
        super().__init__()
        n_qubits = input_size + hidden_size
        self.hidden_size = hidden_size
        self.input_gate = _BatchVQC(vqc_depth, n_qubits, hidden_size)
        self.forget_gate = _BatchVQC(vqc_depth, n_qubits, hidden_size)
        self.cell_gate = _BatchVQC(vqc_depth, n_qubits, hidden_size)
        self.output_gate = _BatchVQC(vqc_depth, n_qubits, hidden_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_t, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x_t, h_prev), dim=1)
        i_t = torch.sigmoid(self.input_gate(combined))
        f_t = torch.sigmoid(self.forget_gate(combined))
        g_t = torch.tanh(self.cell_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        out = self.output_fc(h_t)
        return out, h_t, c_t


class _BatchQLSTMModel(nn.Module):
    """Sequence model that applies the custom QLSTM cell over time."""

    def __init__(self, input_size, hidden_size, output_size, vqc_depth,
                 qlstm_input):
        super().__init__()
        # Project the raw feature vector to a tiny quantum input size.
        self.compress = nn.Linear(input_size, qlstm_input)
        self.hidden_size = hidden_size
        self.cell = _BatchQLSTMCell(
            input_size=qlstm_input,
            hidden_size=hidden_size,
            output_size=output_size,
            vqc_depth=vqc_depth,
        )

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        x = self.compress(x)
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype,
                              device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype,
                              device=x.device)
        else:
            h_t, c_t = hidden

        outputs = []
        for t in range(seq_len):
            out_t, h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            outputs.append(out_t.unsqueeze(1))
        return torch.cat(outputs, dim=1), (h_t, c_t)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BatchQLSTMTrainer:
    """Batch QLSTM trainer using the same dataset contract as other models."""

    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 hidden_size=1, vqc_depth=1, qlstm_input=1, epochs=2,
                 batch_size=128, lr=0.01, hinge=True, time_budget=8.0):
        self.model_name = 'QUANTUM BATCH QLSTM'
        self.epochs = epochs
        self.batch_size = batch_size
        self.hinge = hinge
        self.time_budget = time_budget

        input_size = tra_pv.shape[2]
        self.model = _BatchQLSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1,
            vqc_depth=vqc_depth,
            qlstm_input=qlstm_input,
        ).double()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Convert once so every epoch works on tensors directly.
        tra_x = torch.from_numpy(tra_pv).double()
        tra_y = torch.from_numpy(tra_gt).double()
        self.val_x = torch.from_numpy(val_pv).double()
        self.val_y = torch.from_numpy(val_gt).double()
        self.tes_x = torch.from_numpy(tes_pv).double()
        self.tes_y = torch.from_numpy(tes_gt).double()

        self.tra_x, self.tra_y = self._cap_samples_to_budget(tra_x, tra_y)

        n_qubits = qlstm_input + hidden_size
        print(
            f'[{self.model_name}] {n_qubits} qubits, depth={vqc_depth}, '
            f'compress {input_size}->{qlstm_input}, '
            f'using {self.tra_x.size(0)}/{tra_pv.shape[0]} train samples'
        )

    def _cap_samples_to_budget(self, tra_x, tra_y):
        """Estimate runtime from a pilot pass, then cap train/val/test sizes."""
        pilot_n = min(2, tra_x.size(0))
        self.model.eval()
        with torch.no_grad():
            start = time()
            self.model(tra_x[:pilot_n])
            pilot_sec = time() - start
        sec_per_sample = pilot_sec / max(pilot_n, 1)

        # Split per-epoch budget between fit and evaluation.
        per_epoch_budget = self.time_budget / max(self.epochs, 1)
        train_budget = per_epoch_budget * 0.6
        eval_budget = per_epoch_budget * 0.4

        # Training does forward + backward + optimizer step, so approx x3.
        train_cost = max(sec_per_sample * 3.0, 1e-9)
        max_train = max(int(train_budget / train_cost), pilot_n)
        if max_train < tra_x.size(0):
            keep = torch.randperm(tra_x.size(0))[:max_train]
            tra_x = tra_x[keep]
            tra_y = tra_y[keep]

        # Validation/test are forward-only and should stay aligned for CFA.
        eval_cost = max(sec_per_sample, 1e-9)
        max_eval = max(int((eval_budget / 2.0) / eval_cost), pilot_n)
        if max_eval < self.val_x.size(0):
            self.val_x = self.val_x[:max_eval]
            self.val_y = self.val_y[:max_eval]
        if max_eval < self.tes_x.size(0):
            self.tes_x = self.tes_x[:max_eval]
            self.tes_y = self.tes_y[:max_eval]

        return tra_x, tra_y

    def _eval_perf(self, pred_np, gt_np):
        if self.hinge:
            pred_bin = (pred_np > 0.5).astype(int)
        else:
            pred_bin = np.round(pred_np).astype(int)
        gt_bin = gt_np.ravel().astype(int)
        pred_flat = pred_bin.ravel()
        return {
            'acc': accuracy_score(gt_bin, pred_flat),
            'mcc': matthews_corrcoef(gt_bin, pred_flat),
        }

    def _predict_last_step(self, x):
        self.model.eval()
        with torch.no_grad():
            out, _ = self.model(x)
        return out[:, -1, :].cpu().numpy()

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        perm = torch.randperm(self.tra_x.size(0))
        x = self.tra_x[perm]
        y = self.tra_y[perm]
        for start in range(0, x.size(0), self.batch_size):
            xb = x[start:start + self.batch_size]
            yb = y[start:start + self.batch_size]
            self.optimizer.zero_grad()
            out, _ = self.model(xb)
            loss = self.loss_fn(out[:, -1, :], yb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def train(self, summary=None):
        if summary:
            summary.start_model(self.model_name)
        monitor = SystemMonitor(
            model_name=self.model_name,
            total_epochs=self.epochs,
            summary=summary,
        )

        best_valid_perf = {'acc': 0, 'mcc': -2}
        best_test_perf = {'acc': 0, 'mcc': -2}
        all_val_rows = []
        all_tes_rows = []
        train_t0 = time()

        monitor.start()
        for epoch in range(self.epochs):
            t1 = time()

            monitor.update(f'Training batches (epoch {epoch + 1}/{self.epochs})')
            avg_loss = self._train_one_epoch()
            monitor.log('----->>>>> Training loss: %.6f' % avg_loss)

            monitor.update('Evaluating validation set')
            val_pred = self._predict_last_step(self.val_x)
            val_perf = self._eval_perf(val_pred, self.val_y.cpu().numpy())
            monitor.log('\tVal per: %s' % val_perf)

            monitor.update('Evaluating test set')
            tes_pred = self._predict_last_step(self.tes_x)
            tes_perf = self._eval_perf(tes_pred, self.tes_y.cpu().numpy())
            monitor.log('\tTest per: %s' % tes_perf)

            val_epoch = np.full((val_pred.shape[0], 1), epoch)
            tes_epoch = np.full((tes_pred.shape[0], 1), epoch)
            all_val_rows.append(np.hstack([val_epoch, val_pred, self.val_y.cpu().numpy()]))
            all_tes_rows.append(np.hstack([tes_epoch, tes_pred, self.tes_y.cpu().numpy()]))

            if val_perf['acc'] > best_valid_perf['acc']:
                best_valid_perf = copy.copy(val_perf)
                best_test_perf = copy.copy(tes_perf)

            t4 = time()
            monitor.epoch_done(epoch, t4 - t1)
            monitor.log('epoch: %d time: %.4f' % (epoch, t4 - t1))

        monitor.stop()
        print(f'\n[{self.model_name}] Best Valid performance:', best_valid_perf)
        print(f'\t[{self.model_name}] Best Test performance:', best_test_perf)

        if summary:
            summary.finish_model(
                self.model_name, best_valid_perf, best_test_perf, time() - train_t0
            )
            if all_val_rows:
                summary.save_predictions(self.model_name, 'val', np.vstack(all_val_rows))
                summary.save_predictions(self.model_name, 'test', np.vstack(all_tes_rows))
            summary.save_model(self.model_name, self.model.state_dict())
        return best_valid_perf, best_test_perf
