"""Base trainer for scikit-learn classifiers."""
import copy
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef
from time import time

from .monitor import SystemMonitor


class SklearnTrainer:
    """Generic trainer that wraps any sklearn classifier.

    Auto-calibrates training set size to fit within time_budget.
    """

    def __init__(self, model_name, estimator, tra_pv, tra_gt, val_pv, val_gt,
                 tes_pv, tes_gt, hinge=True, time_budget=0.8):
        self.model_name = model_name
        self.estimator = estimator
        self.hinge = hinge

        # Flatten 3D (samples, seq, features) -> 2D (samples, seq*features)
        tra_x = tra_pv.reshape(tra_pv.shape[0], -1)
        tra_y = tra_gt.ravel()
        self.val_x = val_pv.reshape(val_pv.shape[0], -1)
        self.val_y = val_gt.ravel()
        self.tes_x = tes_pv.reshape(tes_pv.shape[0], -1)
        self.tes_y = tes_gt.ravel()

        # Binarize labels
        if self.hinge:
            self.val_y_cls = (self.val_y > 0.5).astype(int)
            self.tes_y_cls = (self.tes_y > 0.5).astype(int)
        else:
            self.val_y_cls = np.round(self.val_y).astype(int)
            self.tes_y_cls = np.round(self.tes_y).astype(int)

        # Auto-calibrate: pilot fit on small subset, then cap training size
        pilot_n = min(50, tra_x.shape[0])
        pilot_y = (tra_y[:pilot_n] > 0.5).astype(int) if self.hinge else np.round(tra_y[:pilot_n]).astype(int)
        try:
            t0 = time()
            self.estimator.fit(tra_x[:pilot_n], pilot_y)
            pilot_sec = time() - t0
            sec_per_sample = pilot_sec / pilot_n
        except Exception as e:
            # Pilot fit failed (e.g., QDA rank-deficient covariance with
            # fewer samples than features). Use all training data instead.
            from sklearn.base import clone
            self.estimator = clone(estimator)
            print(f'[{model_name}] Pilot fit failed ({e}), using all data')
            sec_per_sample = 0

        # Reserve 30% of budget for predict, use 70% for fit
        fit_budget = time_budget * 0.7
        max_train = max(int(fit_budget / max(sec_per_sample, 1e-9)), pilot_n)

        if max_train < tra_x.shape[0]:
            perm = np.random.RandomState(42).permutation(tra_x.shape[0])[:max_train]
            tra_x = tra_x[perm]
            tra_y = tra_y[perm]

        self.tra_x = tra_x
        self.tra_y = tra_y
        if self.hinge:
            self.tra_y_cls = (self.tra_y > 0.5).astype(int)
        else:
            self.tra_y_cls = np.round(self.tra_y).astype(int)

        print(f'[{model_name}] {type(estimator).__name__}, '
              f'{self.tra_x.shape[1]} features, '
              f'~{sec_per_sample:.6f}s/sample, '
              f'using {self.tra_x.shape[0]}/{tra_pv.shape[0]} train samples')

    def _eval_perf(self, pred_proba, gt_cls):
        binary_pred = (pred_proba > 0.5).astype(int)
        acc = accuracy_score(gt_cls, binary_pred)
        mcc = matthews_corrcoef(gt_cls, binary_pred)
        return {'acc': acc, 'mcc': mcc}

    def train(self, summary=None):
        if summary:
            summary.start_model(self.model_name)
        monitor = SystemMonitor(
            model_name=self.model_name, total_epochs=1, summary=summary)
        best_valid_perf = {'acc': 0, 'mcc': -2}
        best_test_perf = {'acc': 0, 'mcc': -2}
        all_val_rows = []
        all_tes_rows = []
        train_t0 = time()

        monitor.start()

        t1 = time()
        monitor.update('Fitting model')
        self.estimator.fit(self.tra_x, self.tra_y_cls)

        monitor.update('Evaluating validation set')
        if hasattr(self.estimator, 'predict_proba'):
            val_prob = self.estimator.predict_proba(self.val_x)[:, 1]
            tes_prob = self.estimator.predict_proba(self.tes_x)[:, 1]
        else:
            val_prob = self.estimator.decision_function(self.val_x)
            tes_prob = self.estimator.decision_function(self.tes_x)

        val_perf = self._eval_perf(val_prob, self.val_y_cls)
        monitor.log('\tVal per: %s' % val_perf)

        monitor.update('Evaluating test set')
        tes_perf = self._eval_perf(tes_prob, self.tes_y_cls)
        monitor.log('\tTest per: %s' % tes_perf)

        val_pred = val_prob.reshape(-1, 1)
        tes_pred = tes_prob.reshape(-1, 1)
        epoch_col = np.full((val_pred.shape[0], 1), 0)
        all_val_rows.append(np.hstack([epoch_col, val_pred,
                                       self.val_y.reshape(-1, 1)]))
        epoch_col = np.full((tes_pred.shape[0], 1), 0)
        all_tes_rows.append(np.hstack([epoch_col, tes_pred,
                                       self.tes_y.reshape(-1, 1)]))

        best_valid_perf = copy.copy(val_perf)
        best_test_perf = copy.copy(tes_perf)

        t4 = time()
        monitor.epoch_done(0, t4 - t1)
        monitor.log('epoch: 0 time: %.4f' % (t4 - t1))

        monitor.stop()
        print(f'\n[{self.model_name}] Best Valid performance:', best_valid_perf)
        print(f'\t[{self.model_name}] Best Test performance:', best_test_perf)
        if summary:
            summary.finish_model(self.model_name, best_valid_perf,
                                 best_test_perf, time() - train_t0)
            if all_val_rows:
                summary.save_predictions(self.model_name, 'val',
                                         np.vstack(all_val_rows))
                summary.save_predictions(self.model_name, 'test',
                                         np.vstack(all_tes_rows))
            summary.save_model(self.model_name, self.estimator)
        return best_valid_perf, best_test_perf
