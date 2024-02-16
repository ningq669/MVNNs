import torch
import torch.nn as nn
import numpy as np
import re
import random
from datapro import make_data
from train import train_test


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(3701)

    batch_size = 64
    # hidden_size = 64
    d_k = d_v = 32  # dimension of K(=Q), V
    n_layers = 1  # number of Encoder of Decoder Layer
    n_heads = 4  # number of heads in Multi-Head Attention
    initial_lr = 0.001
    # model = BE_sub(54)
    datasets = make_data()
    epoch = 30
    result = train_test(datasets, epoch, batch_size)
