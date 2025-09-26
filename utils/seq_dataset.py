import copy
import math
import numpy as np
import random
from torch.utils.data import Dataset

from utils.aug import Compose, reshape, retype

category_noise = 'Gaussian'  # or 'Laplacian'


def select_noise_indices(num_per_class, ratio, seed, num_classes):
    indices = []

    for i in range(num_classes):
        random.seed(seed)
        noisy_num = int(ratio * num_per_class)
        indices += random.sample(
            list(range(i*num_per_class, (i+1)*num_per_class)), noisy_num)
    return indices


def select_retain_indices(total_num, noise_indices, exclude_noise):
    if exclude_noise > 0:
        return list(set(
                        range(len(total_num))
                    ) - set(
                        random.sample(
                            noise_indices, int(exclude_noise * len(noise_indices)))
                    ))
    else:
        return list(range(len(total_num)))


def add_noise_gaussian(seq, snr, seed):
    seq_len = seq.shape[1]

    signal_power = np.sum(
        seq * seq, axis=1, keepdims=True) / seq_len
    np.random.seed(seed)
    noise = np.random.randn(*seq.shape)
    noise = np.sqrt(
        signal_power / math.pow(10., snr / 10.) /
        (np.sum(noise * noise, axis=1, keepdims=True) / seq_len)
    ) * noise
    return seq + noise


def add_noise_laplacian(seq, snr, seed):
    seq_len = seq.shape[1]

    signal_power = np.sum(
        seq * seq, axis=1, keepdims=True) / seq_len
    np.random.seed(seed)
    noise = np.random.laplace(loc=0.0, scale=1.0, size=(seq.shape[0], seq.shape[1], seq.shape[2]))
    noise = np.sqrt(
        signal_power / math.pow(10., snr / 10.) /
        (np.sum(noise * noise, axis=1, keepdims=True) / seq_len)
    ) * noise
    return seq + noise


class SeqDataset(Dataset):
    def __init__(
            self, df_data, num_per_class=None,
            seq_transform=None,
            mimic_noise=False, seed=None,
            noise_ratio=None, snr=None, exclude_noise=False):
        super(SeqDataset, self).__init__()
        seqs = df_data['seq'].tolist()
        labels = df_data['lb'].tolist()

        seqs = np.stack(seqs, axis=0)
        labels = np.stack(labels, axis=0)
        num_classes = np.unique(labels).shape[0]

        if mimic_noise:
            seqs_noisy = copy.deepcopy(seqs)

            assert noise_ratio is not None
            assert snr is not None
            assert seed is not None
            assert num_per_class == len(df_data)/num_classes

            noise_indices = select_noise_indices(
                num_per_class, noise_ratio, seed, num_classes)

            retain_indices = select_retain_indices(
                df_data, noise_indices, exclude_noise)

            if category_noise == 'Gaussian':
                seqs_added_noise = add_noise_gaussian(seqs_noisy[noise_indices], snr, seed)
            else:
                seqs_added_noise = add_noise_laplacian(seqs_noisy[noise_indices], snr, seed)

            seqs_noisy[noise_indices] = seqs_added_noise

            self.seqs = seqs_noisy[retain_indices]
            self.labels = labels[retain_indices]
        else:
            self.seqs = seqs
            self.labels = labels

        self.seq_transform = Compose(
            reshape, retype) if seq_transform is None else seq_transform

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, item):
        return self.seq_transform(self.seqs[item]), self.labels[item]

