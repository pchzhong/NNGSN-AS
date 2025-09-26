import argparse
import os
from datetime import datetime

from utils.make_log import setlogger
import logging
from utils.train_utils import TrainUtils

datat_dir = {
    'LW': './LW_Dataset/',
    'LAB': './LAB_Dataset/'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Training and test model for fault diagnosis.")

    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--data_name', type=str, choices=['LW', 'LAB'],
                        default='LAB',
                        help='class name for generating dataset')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='the directory of data')
    parser.add_argument('--signal_norm', type=str, choices=['0-1', '-1-1', 'mean-std'], default='mean-std',
                        help='the normalization type for signal sample')
    parser.add_argument('--valid_ratio', type=float, default=0.3, help='the proportion of validation samples')
    parser.add_argument('--test_ratio', type=float, default=0.3,
                        help='the proportion of test samples')
    parser.add_argument('--partial_train', type=float, default=1.,
                        help='what percentage of training samples to use')
    parser.add_argument('--mimic_noise', type=bool, default=True)  # True
    parser.add_argument('--train_noise_ratio', type=float, default=1.)
    parser.add_argument('--train_snr', type=float, default=0)
    parser.add_argument('--noise_seed', type=int, default=1001)
    parser.add_argument('--exclude_noise', type=float, default=0.)  # 0.0, 1.0
    parser.add_argument('--batch_size', type=tuple, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=2048)

    parser.add_argument('--model_name', type=str, choices=['NNGSN_AS'],
                        default='NNGSN_AS')

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--t_max', type=int, default=35)  # 10
    parser.add_argument('--eta_min', type=float, default=1e-4)

    parser.add_argument('--max_epoch', type=int, default=100)  # 200
    parser.add_argument('--print_step', type=int, default=5)

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='the directory to save log and model')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    args.data_dir = datat_dir[args.data_name]

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    t_str = datetime.strftime(datetime.now(), '%y-%m%d-%H-%M-%S')
    if args.mimic_noise:
        noise_config = f'Mimic-ratio={args.train_noise_ratio}-snr={args.train_snr}-seed={args.seed}'
    else:
        noise_config = 'None'
    sub_dir = f'Denoise-{args.data_name}_Gaussian_{args.model_name}_noise={noise_config}_{t_str}'
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'main.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info(f'{k}: {v}')

    trainer = TrainUtils(args, save_dir)
    trainer.run()
