import numpy as np
import random
from scipy.signal import resample


class Compose(object):
    def __init__(self, *args):
        self.transforms = [t for t in args]

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


def reshape(seq):
    # print(seq.shape)
    return seq.transpose()


def normalize(seq, norm_type='0-1'):
    if norm_type == '0-1':
        return (seq - seq.min()) / (seq.max() - seq.min())
    elif norm_type == '-1-1':
        return (seq - seq.min()) / (seq.max() - seq.min()) * 2 - 1
    elif norm_type == 'mean-std':
        return (seq - seq.mean()) / seq.std()
    else:
        raise NameError("This normalization is not included!")


def random_add_gaussian(seq, sigma=0.01):
    if np.random.randint(2):
        return seq
    else:
        return seq + np.random.normal(loc=0, scale=sigma, size=seq.shape)


def random_scale(seq, sigma=0.01):
    if np.random.randint(2):
        return seq
    else:
        scale_factor = np.random.normal(loc=1, scale=sigma, size=(seq.shape[0], 1))
        scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
        return seq * scale_matrix


def random_stretch(seq, sigma=0.3):
    if np.random.randint(2):
        return seq
    else:
        seq_aug = np.zeros(seq.shape)
        len = seq.shape[1]
        length = int((sigma * (random.random() - 0.5) + 1) * len)
        for i in range(seq.shape[0]):
            y = resample(seq[i, :], length)
            if length < len:
                if random.random() < 0.5:
                    seq_aug[i, :length] = y
                else:
                    seq_aug[i, len - length:] = y
            else:
                if random.random() < 0.5:
                    seq_aug[i, :] = y[:len]
                else:
                    seq_aug[i, :] = y[length - len:]
        return seq_aug


def random_crop(seq, crop_len=20):
    if np.random.randint(2):
        return seq
    else:
        max_index = seq.shape[1] - crop_len
        random_index = np.random.randint(max_index)
        seq[:, random_index: random_index + crop_len] = 0
        return seq


def retype(seq):
    return seq.astype(np.float32)

