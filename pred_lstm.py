import argparse
import os
import signal
import pathlib
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# Auto-detect pip-installed NVIDIA CUDA libs for GPU support
_nvidia_dir = pathlib.Path(__file__).resolve().parent / '.venv' / 'lib'
for _lib in _nvidia_dir.rglob('nvidia/*/lib'):
    if _lib.is_dir():
        os.environ['LD_LIBRARY_PATH'] = str(_lib) + ':' + os.environ.get('LD_LIBRARY_PATH', '')

from models import AWLSTM, QuantumTrainer, RunSummary

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
