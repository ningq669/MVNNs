import numpy as np

import torch
import csv
import torch.utils.data.dataset as Dataset


def make_data():
    dataset = dict()

    train_p_string = np.loadtxt('./train_p.txt', dtype=np.str_, delimiter='\n')
    train_n_string = np.loadtxt('./train_n.txt', dtype=np.str_, delimiter='\n')

    # test_pos_edges = test_pos_edges.T
    # test_neg_edges = test_neg_edges.T
    train_true_label = np.hstack((np.ones(train_p_string.shape[0]), np.zeros(train_n_string.shape[0])))
    train_true_label = np.array(train_true_label, dtype='float32')
    train_seqence = np.hstack((train_p_string, train_n_string))

    valid_p_string = np.loadtxt('./validation_p.txt', dtype=np.str_, delimiter='\n')
    valid_n_string = np.loadtxt('./validation_n.txt', dtype=np.str_, delimiter='\n')
    valid_true_label = np.hstack((np.ones(valid_p_string.shape[0]), np.zeros(valid_n_string.shape[0])))
    valid_true_label = np.array(valid_true_label, dtype='float32')
    valid_seqence = np.hstack((valid_p_string, valid_n_string))

    test_p_string = np.loadtxt('./test_p.txt', dtype=np.str_, delimiter='\n')
    test_n_string = np.loadtxt('./test_n.txt', dtype=np.str_, delimiter='\n')
    test_true_label = np.hstack((np.ones(test_p_string.shape[0]), np.zeros(test_n_string.shape[0])))
    test_true_label = np.array(test_true_label, dtype='float32')
    test_seqence = np.hstack((test_p_string, test_n_string))

    dataset['train_seq'] = train_seqence
    dataset['train_label'] = train_true_label

    dataset['valid_seq'] = valid_seqence
    dataset['valid_label'] = valid_true_label

    dataset['test_seq'] = test_seqence
    dataset['test_label'] = test_true_label

    return dataset


def make_alldata():
    dataset = dict()

    train_allp = np.loadtxt('./feature/positive.txt', dtype=np.str_, delimiter='\n')
    train_alln = np.loadtxt('./feature/negative.txt', dtype=np.str_, delimiter='\n')

    rng = np.random.default_rng(seed=42)
    neg_samples_shuffled = rng.permutation(train_alln, axis=0)[:train_allp.shape[0]]

    train_all_label = np.hstack((np.ones(train_allp.shape[0]), np.zeros(train_allp.shape[0])))
    train_all_label = np.array(train_all_label, dtype='float32')
    train_all_seqence = np.hstack((train_allp, neg_samples_shuffled))

    dataset['train_all_seq'] = train_all_seqence
    dataset['train_label'] = train_all_label

    return dataset


class SeqDataset(Dataset.Dataset):
    def __init__(self, seqences, labels):
        self.Data = seqences
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label
