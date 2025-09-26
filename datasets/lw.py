import numpy as np
import pandas as pd

import os
import re

import sys
sys.path.append('..')
from utils.seq_dataset import SeqDataset
from utils.aug import Compose, reshape, normalize, \
    random_add_gaussian, random_scale, random_stretch, random_crop, retype


seq_len = 1024
shift_size = 1024
category = np.arange(4)
k = 0
num_feature = 9


def load_txt(data_dir):
    f = open(data_dir, 'r')
    sourceInLine = f.readlines()

    dataset = []
    for line in sourceInLine:
        temp1 = line.strip('\n')
        temp2 = temp1.split('\t')
        dataset.append(temp2)

    dataset1 = dataset[1:85000]
    all_signal = []
    for n_f in range(0, num_feature):
        dataset2 = []
        for i in range(0, len(dataset1)):
            dataset2.append(float(dataset1[i][n_f+k]))
        dataset3 = np.array(dataset2)

        all_signal.append(dataset3)

    all_signal = np.stack(all_signal, axis=0)
    all_signal1 = all_signal.transpose()
    return all_signal1


def signal_train_test_spilt(raw, valid_ratio=0.2, test_ratio=0.2, partial_train=1.):
    """
    Split single raw signal to train and test sequence samples.
    """
    rl = len(raw)
    sl = seq_len
    r_tst = test_ratio
    r_vld = valid_ratio
    ss = shift_size

    n_test = np.floor((rl - sl + ss) * r_tst /
                      ((1 - r_tst - r_vld) * ss + (r_tst + r_vld) * sl)).astype(np.int32)
    n_valid = np.floor(n_test / r_tst * r_vld).astype(np.int32)
    n_train = np.floor(n_test / r_tst - n_test - n_valid).astype(np.int32)
    train_len = (n_train - 1) * ss + sl

    n_train = np.floor(n_train * partial_train).astype(np.int32)

    tr_stride = ss
    vl_stride = sl
    te_stride = sl

    tr_seq, _ = get_seqs(raw, n_train, tr_stride, start=0)
    vl_seq, tr_vl_len = get_seqs(raw, n_valid, vl_stride, start=train_len)
    te_seq, total_len = get_seqs(raw, n_test, te_stride, start=tr_vl_len)

    min_num_train = 32
    min_num_valid = 24
    min_num_test = 24

    train_seq1 = tr_seq[:min_num_train]
    valid_seq1 = vl_seq[:min_num_valid]
    test_seq1 = te_seq[:min_num_test]

    return train_seq1, valid_seq1, test_seq1


def get_seqs(raw, n, stride, start=0):
    """Get sequence samples from raw signal."""
    seqs = []
    end = start + seq_len
    for i in range(n):
        seqs.append(raw[start: end, :])
        start += stride
        end += stride
    return seqs, end - stride


def get_train_test_from_files(data_dir, valid_ratio, test_ratio, partial_train):

    train_seq, train_lb = [], []
    valid_seq, valid_lb = [], []
    test_seq, test_lb = [], []

    datasetname = os.listdir(data_dir)
    datasetname.sort(
        key=lambda tn: int(re.match('^LW(\d+)', tn).group(1)))

    for i in range(len(category)):
        dir_lw = os.path.join(data_dir, datasetname[i], '1500-0.txt')
        signal = load_txt(dir_lw)

        tr_seq, vl_seq, te_seq = signal_train_test_spilt(signal,
                                                         valid_ratio=valid_ratio, test_ratio=test_ratio,
                                                         partial_train=partial_train)

        lb = category[(i % 10)]
        tr_lb = np.repeat(lb, len(tr_seq)).tolist()
        vl_lb = np.repeat(lb, len(vl_seq)).tolist()
        te_lb = np.repeat(lb, len(te_seq)).tolist()

        for total, single in zip([train_seq, valid_seq, test_seq,
                                  train_lb, valid_lb, test_lb],
                                 [tr_seq, vl_seq, te_seq,
                                  tr_lb, vl_lb, te_lb]):
            total.append(single)

        dir_lw = os.path.join(data_dir, datasetname[i], '1500-2.txt')
        signal = load_txt(dir_lw)

        tr_seq, vl_seq, te_seq = signal_train_test_spilt(signal,
                                                         valid_ratio=valid_ratio, test_ratio=test_ratio,
                                                         partial_train=partial_train)

        lb = category[(i % 10)]
        tr_lb = np.repeat(lb, len(tr_seq)).tolist()
        vl_lb = np.repeat(lb, len(vl_seq)).tolist()
        te_lb = np.repeat(lb, len(te_seq)).tolist()

        for total, single in zip([train_seq, valid_seq, test_seq,
                                  train_lb, valid_lb, test_lb],
                                 [tr_seq, vl_seq, te_seq,
                                  tr_lb, vl_lb, te_lb]):
            total.append(single)

        dir_lw = os.path.join(data_dir, datasetname[i], '1500-4.txt')
        signal = load_txt(dir_lw)

        tr_seq, vl_seq, te_seq = signal_train_test_spilt(signal,
                                                         valid_ratio=valid_ratio, test_ratio=test_ratio,
                                                         partial_train=partial_train)

        lb = category[(i % 10)]
        tr_lb = np.repeat(lb, len(tr_seq)).tolist()
        vl_lb = np.repeat(lb, len(vl_seq)).tolist()
        te_lb = np.repeat(lb, len(te_seq)).tolist()

        for total, single in zip([train_seq, valid_seq, test_seq,
                                  train_lb, valid_lb, test_lb],
                                 [tr_seq, vl_seq, te_seq,
                                  tr_lb, vl_lb, te_lb]):
            total.append(single)

        dir_lw = os.path.join(data_dir, datasetname[i], '1500-6.txt')
        signal = load_txt(dir_lw)

        tr_seq, vl_seq, te_seq = signal_train_test_spilt(signal,
                                                         valid_ratio=valid_ratio, test_ratio=test_ratio,
                                                         partial_train=partial_train)

        lb = category[(i % 10)]
        tr_lb = np.repeat(lb, len(tr_seq)).tolist()
        vl_lb = np.repeat(lb, len(vl_seq)).tolist()
        te_lb = np.repeat(lb, len(te_seq)).tolist()

        for total, single in zip([train_seq, valid_seq, test_seq,
                                  train_lb, valid_lb, test_lb],
                                 [tr_seq, vl_seq, te_seq,
                                  tr_lb, vl_lb, te_lb]):
            total.append(single)

        dir_lw = os.path.join(data_dir, datasetname[i], '1500-8.txt')
        signal = load_txt(dir_lw)

        tr_seq, vl_seq, te_seq = signal_train_test_spilt(signal,
                                                         valid_ratio=valid_ratio, test_ratio=test_ratio,
                                                         partial_train=partial_train)

        lb = category[(i % 10)]
        tr_lb = np.repeat(lb, len(tr_seq)).tolist()
        vl_lb = np.repeat(lb, len(vl_seq)).tolist()
        te_lb = np.repeat(lb, len(te_seq)).tolist()

        for total, single in zip([train_seq, valid_seq, test_seq,
                                  train_lb, valid_lb, test_lb],
                                 [tr_seq, vl_seq, te_seq,
                                  tr_lb, vl_lb, te_lb]):
            total.append(single)

        dir_lw = os.path.join(data_dir, datasetname[i], '1500-10.txt')
        signal = load_txt(dir_lw)

        tr_seq, vl_seq, te_seq = signal_train_test_spilt(signal,
                                                         valid_ratio=valid_ratio, test_ratio=test_ratio,
                                                         partial_train=partial_train)

        lb = category[(i % 10)]
        tr_lb = np.repeat(lb, len(tr_seq)).tolist()
        vl_lb = np.repeat(lb, len(vl_seq)).tolist()
        te_lb = np.repeat(lb, len(te_seq)).tolist()

        for total, single in zip([train_seq, valid_seq, test_seq,
                                  train_lb, valid_lb, test_lb],
                                 [tr_seq, vl_seq, te_seq,
                                  tr_lb, vl_lb, te_lb]):
            total.append(single)

    min_num_train = 192
    min_num_valid = 144
    min_num_test = 144

    train_seq1, train_lb1 = [], []
    valid_seq1, valid_lb1 = [], []
    test_seq1, test_lb1 = [], []
    for i in range(len(category)*6):

        train_seq1 += train_seq[i]
        train_lb1 += train_lb[i]
        valid_seq1 += valid_seq[i]
        valid_lb1 += valid_lb[i]
        test_seq1 += test_seq[i]
        test_lb1 += test_lb[i]

    train_seq = train_seq1
    train_lb = train_lb1
    valid_seq = valid_seq1
    valid_lb = valid_lb1
    test_seq = test_seq1
    test_lb = test_lb1

    train_df = pd.DataFrame({'seq': train_seq, 'lb': train_lb})
    valid_df = pd.DataFrame({'seq': valid_seq, 'lb': valid_lb})
    test_df = pd.DataFrame({'seq': test_seq, 'lb': test_lb})

    return train_df, valid_df, test_df, min_num_train, min_num_valid, min_num_test


def data_transforms(dataset_type, norm_type):
    transform = {
        'train': Compose(
            reshape,
            random_add_gaussian,
            random_scale,
            random_stretch,
            random_crop,
            lambda x: normalize(x, norm_type),
            retype),
        'valid': Compose(
            reshape,
            lambda x: normalize(x, norm_type),
            retype),
        'test': Compose(
            reshape,
            lambda x: normalize(x, norm_type),
            retype)}
    return transform[dataset_type]


class LW(object):
    num_in_chns = 9
    num_classes = len(category)

    def __init__(self, data_dir, signal_norm='-1-1'):
        self.data_dir = data_dir
        self.signal_norm = signal_norm

    def data_prepare(
            self,
            valid_ratio=0.2, test_ratio=0.2,
            partial_train=1.,
            mimic_noise=True, noise_seed=1665,
            train_noise_ratio=1., train_snr=8., exclude_noise=1.0):

        train_df, valid_df, test_df, num_per_class_train, num_per_class_valid, num_per_class_test = get_train_test_from_files(
            self.data_dir,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            partial_train=partial_train)

        train_set = SeqDataset(
            df_data=train_df,
            num_per_class=num_per_class_train,
            seq_transform=data_transforms(
                'train', norm_type=self.signal_norm),
            mimic_noise=mimic_noise, seed=noise_seed,
            noise_ratio=train_noise_ratio, snr=train_snr, exclude_noise=exclude_noise)
        valid_set = SeqDataset(
            df_data=valid_df,
            num_per_class=num_per_class_valid,
            seq_transform=data_transforms(
                'valid', norm_type=self.signal_norm),
            mimic_noise=mimic_noise, seed=noise_seed,
            noise_ratio=train_noise_ratio, snr=train_snr, exclude_noise=exclude_noise)
        test_set = SeqDataset(
            df_data=test_df,
            num_per_class=num_per_class_test,
            seq_transform=data_transforms(
                'test', norm_type=self.signal_norm),
            mimic_noise=mimic_noise, seed=noise_seed,
            noise_ratio=train_noise_ratio, snr=train_snr, exclude_noise=exclude_noise)

        return train_set, valid_set, test_set

