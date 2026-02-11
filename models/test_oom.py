"""TEST MODEL: Deliberately allocates excessive RAM to test memory limits.

This is NOT a real model. It exists solely to verify that the MemoryGuard
kills models that exceed the per-model RAM budget.
"""
import copy
import numpy as np
from time import time

from .monitor import SystemMonitor


class OOMTestTrainer:
    """Test trainer that deliberately allocates too much RAM."""

    def __init__(self, tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
                 epochs=1, batch_size=256, hinge=True, target_gb=2.0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.hinge = hinge
        self.target_gb = target_gb

        self.tra_pv = tra_pv
        self.tra_gt = tra_gt
        self.val_pv = val_pv
        self.val_gt = val_gt
        self.tes_pv = tes_pv
        self.tes_gt = tes_gt

        print(f'[TEST OOM] Will attempt to allocate ~{target_gb:.1f} GB '
              f'(should be killed by MemoryGuard)')

    def train(self, summary=None):
        if summary:
            summary.start_model('TEST OOM')
        monitor = SystemMonitor(
            model_name='TEST OOM', total_epochs=self.epochs, summary=summary)
        best_valid_perf = {'acc': 0, 'mcc': -2}
        best_test_perf = {'acc': 0, 'mcc': -2}
        all_val_rows = []
        all_tes_rows = []
        train_t0 = time()

        monitor.start()
        for epoch in range(self.epochs):
            t1 = time()
            monitor.update(f'Allocating memory (epoch {epoch}/{self.epochs})')

            # ---- Deliberately hog RAM ----
            hog = []
            chunk_mb = 128
            n_chunks = int(self.target_gb * 1024 / chunk_mb)
            for i in range(n_chunks):
                monitor.log(f'[TEST OOM] Allocating chunk {i+1}/{n_chunks} '
                            f'({chunk_mb} MB each)...')
                # 128 MB per chunk (16M float64 = 128 MB)
                hog.append(np.ones(chunk_mb * 1024 * 1024 // 8, dtype=np.float64))

            # Random predictions (won't normally reach here)
            monitor.update('Evaluating validation set')
            val_pred = np.random.rand(self.val_gt.shape[0], 1)
            val_perf = {'acc': 0.5, 'mcc': 0.0}
            monitor.log('\tVal per: %s' % val_perf)

            monitor.update('Evaluating test set')
            tes_pred = np.random.rand(self.tes_gt.shape[0], 1)
            tes_perf = {'acc': 0.5, 'mcc': 0.0}
            monitor.log('\tTest per: %s' % tes_perf)

            epoch_col = np.full((val_pred.shape[0], 1), epoch)
            all_val_rows.append(np.hstack([epoch_col, val_pred, self.val_gt]))
            epoch_col = np.full((tes_pred.shape[0], 1), epoch)
            all_tes_rows.append(np.hstack([epoch_col, tes_pred, self.tes_gt]))

            if val_perf['acc'] > best_valid_perf['acc']:
                best_valid_perf = copy.copy(val_perf)
                best_test_perf = copy.copy(tes_perf)

            t4 = time()
            monitor.epoch_done(epoch, t4 - t1)
            monitor.log('epoch: %d time: %.4f' % (epoch, t4 - t1))

            del hog  # free for next epoch

        monitor.stop()
        print('\n[TEST OOM] Best Valid performance:', best_valid_perf)
        print('\t[TEST OOM] Best Test performance:', best_test_perf)
        if summary:
            summary.finish_model('TEST OOM', best_valid_perf, best_test_perf,
                                 time() - train_t0)
            if all_val_rows:
                summary.save_predictions('TEST OOM', 'val',
                                         np.vstack(all_val_rows))
                summary.save_predictions('TEST OOM', 'test',
                                         np.vstack(all_tes_rows))
        return best_valid_perf, best_test_perf
