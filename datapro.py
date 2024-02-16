import numpy as np
import torch.utils.data.dataset as Dataset




def make_data():
    dataset = dict()

    train_seqs = np.loadtxt('./feature/train_seq.csv', dtype=np.str_, delimiter='\n')
    train_labels = np.loadtxt('./feature/train_label.csv', dtype=np.float32)

    valid_seqs = np.loadtxt('./feature/valid_seq.csv', dtype=np.str_, delimiter='\n')
    valid_labels = np.loadtxt('./feature/valid_label.csv', dtype=np.float32)

    test_seqs = np.loadtxt('./feature/test_seq.csv', dtype=np.str_, delimiter='\n')
    test_labels = np.loadtxt('./feature/test_label.csv', dtype=np.float32)



    dataset['train_seq'] = train_seqs
    dataset['train_label'] = train_labels

    dataset['valid_seq'] = valid_seqs
    dataset['valid_label'] = valid_labels

    dataset['test_seq'] = test_seqs
    dataset['test_label'] = test_labels

    return dataset



def make_alldata():
    dataset = dict()

    train_allp = np.loadtxt('./feature/positive.txt', dtype=np.str_, delimiter='\n')
    train_alln = np.loadtxt('./feature/negative.txt', dtype=np.str_, delimiter='\n')


    rng = np.random.default_rng(seed=5555)
    neg_samples_shuffled = rng.permutation(train_alln, axis=0)[:train_allp.shape[0]]

    train_all_label = np.hstack((np.ones(train_allp.shape[0]), np.zeros(train_allp.shape[0])))
    train_all_label = np.array(train_all_label, dtype='float32')
    train_all_seqence = np.hstack((train_allp, neg_samples_shuffled))

    dataset['train_all_seq'] = train_all_seqence
    dataset['train_label'] = train_all_label

    return dataset


class SeqDataset(Dataset.Dataset):
    def __init__(self,seqences,labels):
        self.Data = seqences
        self.Label = labels
        a = 1

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data,label




