import copy
from datetime import datetime
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error
from sklearn.utils import shuffle
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from time import time

from .monitor import SystemMonitor


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

        all_val_rows = []
        all_tes_rows = []
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

            epoch_col = np.full((val_pre.shape[0], 1), i)
            all_val_rows.append(np.hstack([epoch_col, val_pre, self.val_gt]))
            epoch_col = np.full((tes_pre.shape[0], 1), i)
            all_tes_rows.append(np.hstack([epoch_col, tes_pre, self.tes_gt]))

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
            if all_val_rows:
                summary.save_predictions('CLASSICAL ALSTM', 'val', np.vstack(all_val_rows))
                summary.save_predictions('CLASSICAL ALSTM', 'test', np.vstack(all_tes_rows))
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
