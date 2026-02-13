import copy
import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score, matthews_corrcoef
from time import time
import torch
import torch.nn as nn

from .monitor import SystemMonitor


# ---------------------------------------------------------------------------
# Quantum circuit helpers
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


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

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
        # Use first-N (not random) so indices align with other models for CFA
        max_eval = max(int(eval_budget / (sec_per_sample * 2)), pilot_n)
        if max_eval < self.val_x.size(0):
            self.val_x = self.val_x[:max_eval]
            self.val_y = self.val_y[:max_eval]
        if max_eval < self.tes_x.size(0):
            self.tes_x = self.tes_x[:max_eval]
            self.tes_y = self.tes_y[:max_eval]

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
        all_val_rows = []
        all_tes_rows = []
        train_t0 = time()
        last_checkpoint = time()
        CHECKPOINT_INTERVAL = 300  # save every 5 minutes

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

                # Time-based checkpoint during long epochs
                if summary and (time() - last_checkpoint) >= CHECKPOINT_INTERVAL:
                    summary.save_model('QUANTUM LSTM', self.model.state_dict())
                    last_checkpoint = time()
                    monitor.log('[QUANTUM LSTM] Checkpoint saved (%.0fs elapsed)' % (last_checkpoint - train_t0))
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

            epoch_col = np.full((val_pred.shape[0], 1), epoch)
            all_val_rows.append(np.hstack([epoch_col, val_pred, self.val_y.numpy()]))
            epoch_col = np.full((tes_pred.shape[0], 1), epoch)
            all_tes_rows.append(np.hstack([epoch_col, tes_pred, self.tes_y.numpy()]))

            if val_perf['acc'] > best_valid_perf['acc']:
                best_valid_perf = copy.copy(val_perf)
                best_test_perf = copy.copy(tes_perf)

            t4 = time()
            monitor.epoch_done(epoch, t4 - t1)
            monitor.log('epoch: %d time: %.4f' % (epoch, t4 - t1))

            # Checkpoint after every epoch so progress is never lost
            if summary:
                summary.save_model('QUANTUM LSTM', self.model.state_dict())
                if all_val_rows:
                    summary.save_predictions('QUANTUM LSTM', 'val', np.vstack(all_val_rows))
                    summary.save_predictions('QUANTUM LSTM', 'test', np.vstack(all_tes_rows))

        monitor.stop()
        print('\n[QUANTUM LSTM] Best Valid performance:', best_valid_perf)
        print('\t[QUANTUM LSTM] Best Test performance:', best_test_perf)
        if summary:
            summary.finish_model('QUANTUM LSTM', best_valid_perf, best_test_perf, time() - train_t0)
        return best_valid_perf, best_test_perf
