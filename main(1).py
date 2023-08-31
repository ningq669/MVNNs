import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn import preprocessing
from tqdm import tqdm
from sklearn import metrics
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, matthews_corrcoef
from collections import Counter
import re
import random


class New_Attention(nn.Module):
    def __init__(self, inputsize, hidden_size):
        super(New_Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=inputsize, num_heads=4, batch_first=True)  ##设置一下
        self.wq = nn.Linear(inputsize, hidden_size, bias=False)
        self.wk = nn.Linear(inputsize, hidden_size, bias=False)
        self.wv = nn.Linear(inputsize, hidden_size, bias=False)

    def forward(self, x):
        attn_output, attn_output_weights = self.attention(x, x, x)

        return attn_output


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiHeadAttention, self).__init__()

        self.dim = d_model
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size1 = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size1, -1, n_heads, d_k).transpose(1,
                                                                            2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size1, -1, n_heads, d_k).transpose(1,
                                                                            2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size1, -1, n_heads, d_v).transpose(1,
                                                                            2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size1, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.dim)(output + residual)


class AtentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AtentionLayer, self).__init__()
        self.dim = d_model
        self.enc_self_attn = MultiHeadAttention(self.dim)

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len, d_model]
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.dim = d_model
        self.layers = nn.ModuleList([AtentionLayer(self.dim) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len, d_model]
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model]
            enc_outputs = layer(enc_inputs)
        return enc_outputs


class CNN(nn.Module):
    def __init__(self, inputsize, outputsize, kernelsize, step):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inputsize, out_channels=outputsize, kernel_size=kernelsize, stride=step, padding=1),
            nn.BatchNorm2d(outputsize),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class BLSTM(nn.Module):
    def __init__(self, input_size_lstm, hidden_size, layer_num):
        super(BLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size_lstm, hidden_size=hidden_size, num_layers=layer_num,
                            bidirectional=True, batch_first=True)

    def forward(self, X):
        batch_size_X = X.shape[0]
        input_X = X.transpose(0, 1)

        # hidden_state_X = torch.zeros(1 * 2, batch_size_X, hidden_size)
        # cell_state_X = torch.zeros(1 * 2, batch_size_X, hidden_size)

        outputs_lstm, h_n = self.lstm(input_X)

        return outputs_lstm


class BGRU(nn.Module):
    def __init__(self, input_size_gru, hidden_size, layer_num):
        super(BGRU, self).__init__()

        self.gru = nn.GRU(input_size=input_size_gru, hidden_size=hidden_size, num_layers=layer_num, bidirectional=True,
                          batch_first=True)

    def forward(self, X):
        # batch_size_X = X.shape[0]
        input_X = X.transpose(0, 1)

        # hidden_state_X = torch.zeros(1 * 2, batch_size_X, hidden_size)##?多余写
        # cell_state_X = torch.zeros(1 * 2, batch_size_X, hidden_size)

        outputs_gru, h_n = self.gru(input_X)

        return outputs_gru


class FCNN(nn.Module):
    def __init__(self, input_sizes, drop):
        super(FCNN, self).__init__()

        self.dropout = nn.Dropout(p=drop)
        self.fc1 = nn.Linear(input_sizes, 64)
        self.fc2 = nn.Linear(64, 1)
        # self.softsign = nn.Softsign()
        self.relu = nn.ReLU()
        # dropout = nn.Dropout()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        outputs = self.fc2(x)
        outputs = self.sigmoid(outputs)

        return outputs


def BE(gene):
    with open("BE.txt") as f:
        records = f.readlines()[1:]
    BE = []
    for i in records:
        array = i.rstrip().split() if i.rstrip() != '' else None
        BE.append(array)
    BE = np.array(
        [float(BE[i][j]) for i in range(len(BE)) for j in range(len(BE[i]))]).reshape((20, 21))
    BE = BE.transpose()
    AA = 'ACDEFGHIKLMNPQRSTWYV*'
    GENE_BE = {}
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)
    gene_array = np.zeros((n, 20))
    for i in range(n):
        gene_array[i] = BE[(GENE_BE[gene[i]])]
    return gene_array


def BLOSUM62(gene):
    with open("blosum62.txt") as f:
        records = f.readlines()[1:]
    blosum62 = []
    for i in records:
        array = i.rstrip().split() if i.rstrip() != '' else None
        blosum62.append(array)
    blosum62 = np.array(
        [float(blosum62[i][j]) for i in range(len(blosum62)) for j in range(len(blosum62[i]))]).reshape((20, 21))
    blosum62 = blosum62.transpose()
    GENE_BE = {}
    AA = 'ARNDCQEGHILKMFPSTWYV*'
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)
    gene_array = np.zeros((n, 20))
    for i in range(n):
        gene_array[i] = blosum62[(GENE_BE[gene[i]])]
    return gene_array


def AAI(gene):
    with open("AAindex.txt") as f:
        records = f.readlines()[1:]
    AAI = []
    for i in records:
        array = i.rstrip().split()[1:] if i.rstrip() != '' else None
        AAI.append(array)
    AAI = np.array(
        [float(AAI[i][j]) for i in range(len(AAI)) for j in range(len(AAI[i]))]).reshape((531, 21))
    AAI = AAI.transpose()
    GENE_BE = {}
    AA = 'ACDEFGHIKLMNPQRSTWYV*'
    for i in range(len(AA)):
        GENE_BE[AA[i]] = i
    n = len(gene)
    gene_array = np.zeros((n, 531))
    for i in range(n):
        gene_array[i] = AAI[(GENE_BE[gene[i]])]

    return gene_array


def CKSAAP(gene, gap=5):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    sequence = gene
    code = []
    for g in range(gap + 1):
        myDict = {}
        for pair in aaPairs:
            myDict[pair] = 0
        sum = 0
        for index1 in range(len(sequence)):
            index2 = index1 + g + 1
            if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                index2] in AA:
                myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                sum = sum + 1
        for pair in aaPairs:
            code.append(myDict[pair] / sum)

    final_code = np.array(code)
    # size_c = len(final_code)

    return final_code


def EAAC(gene, window=5):
    AA = 'ACDEFGHIKLMNPQRSTVWY'

    sequence = gene.replace("*", "-")
    code = []
    for j in range(len(sequence)):
        if j < len(sequence) and j + window <= len(sequence):
            count = Counter(re.sub('-', '', sequence[j:j + window]))
            for key in count:
                count[key] = count[key] / len(re.sub('-', '', sequence[j:j + window]))
            for aa in AA:
                code.append(count[aa])
    final_code = np.array(code)

    return final_code


class BE_sub(nn.Module):
    def __init__(self, inputsize):
        super(BE_sub, self).__init__()

        self.BE_gru = BGRU(inputsize, 64, 7)
        self.dropout = nn.Dropout(p=0.5)
        self.BE_fcnn = FCNN(31 * 64 * 2, 0.0)

    def forward(self, inputs_p, inputs_n):
        inputs = []
        inputs.extend(inputs_p)
        inputs.extend(inputs_n)
        # print(inputs)
        n = len(inputs)
        m_l = len(inputs[0])
        x_BE = np.zeros((n, m_l, 20))

        maxabs = preprocessing.MinMaxScaler()
        for i in range(n):
            x_BE[i] = BE(inputs[i])
            x_BE[i] = maxabs.fit_transform(x_BE[i])
        cor_feature = torch.tensor(x_BE, dtype=torch.float32, requires_grad=True)

        batch_size = cor_feature.shape[0]
        x_o = self.BE_gru(cor_feature)  # [seq,batch_size,hidden_size * 2]
        x_o = self.dropout(x_o)
        x_o = x_o.permute(1, 0, 2)

        x_f = x_o.reshape(batch_size, -1)

        outputs = self.BE_fcnn(x_f)

        return outputs


class Blosum62_sub(nn.Module):
    def __init__(self, inputsize):
        super(Blosum62_sub, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.Blosum62_lstm = BLSTM(inputsize, 64, 3)
        self.Blosum62_att = New_Attention(64 * 2, 32)
        self.Blosum62_fcnn = FCNN(31 * 128, 0.0)

    def forward(self, inputs_p, inputs_n):
        inputs = []
        inputs.extend(inputs_p)
        inputs.extend(inputs_n)
        # print(inputs)
        n = len(inputs)
        m_l = len(inputs[0])
        x_62 = np.zeros((n, m_l, 20))

        maxabs = preprocessing.MinMaxScaler()
        for i in range(n):
            x_62[i] = BLOSUM62(inputs[i])
            x_62[i] = maxabs.fit_transform(x_62[i])

        cor_feature = torch.tensor(x_62, dtype=torch.float32, requires_grad=True)

        batch_size = cor_feature.shape[0]

        x_o = self.Blosum62_lstm(cor_feature)  # [seq,batch_size,hidden_size * 2]
        x_o = self.dropout(x_o)
        x_attention_in = x_o.permute(1, 0, 2)

        x_no = self.Blosum62_att(x_attention_in)
        x_no = self.dropout(x_no)
        x_f = x_no.reshape(batch_size, -1)
        # print(flatten.size())
        outputs = self.Blosum62_fcnn(x_f)

        return outputs


class AAI_sub(nn.Module):
    def __init__(self, inputsize):
        super(AAI_sub, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.AAI_gru = BGRU(inputsize, 256, 3)
        self.AAI_att = New_Attention(256 * 2, 32)
        self.AAI_fcnn = FCNN(31 * 512, 0.0)

    def forward(self, inputs_p, inputs_n):
        inputs = []
        inputs.extend(inputs_p)
        inputs.extend(inputs_n)
        # print(inputs)
        n = len(inputs)
        m_l = len(inputs[0])

        x_AAI = np.zeros((n, m_l, 531))
        maxabs = preprocessing.MinMaxScaler()
        for i in range(n):
            x_AAI[i] = AAI(inputs[i])
            x_AAI[i] = maxabs.fit_transform(x_AAI[i])

        cor_feature = torch.tensor(x_AAI, dtype=torch.float32, requires_grad=True)

        batch_size = cor_feature.shape[0]

        x_o = self.AAI_gru(cor_feature)  # [seq,batch_size,hidden_size * 2]
        x_o = self.dropout(x_o)
        x_attention_in = x_o.permute(1, 0, 2)  ##看看对不对

        x_no = self.AAI_att(x_attention_in)
        x_no = self.dropout(x_no)
        # x_o = self.AAI_att(x_o)
        x_f = x_no.reshape(batch_size, -1)
        # print(flatten.size())
        outputs = self.AAI_fcnn(x_f)

        return outputs


class CKSAAP_sub(nn.Module):
    def __init__(self, inputsize):
        super(CKSAAP_sub, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.CKSAAP_cnn1 = CNN(inputsize, 64, 3, 2)  ##看看input_size
        self.CKSAAP_cnn2 = CNN(64, 64, 3, 2)
        self.mp = nn.MaxPool2d(stride=2, kernel_size=3)
        self.CKSAAP_cnn3 = CNN(64, 64, 3, 1)
        self.CKSAAP_cnn4 = CNN(64, 128, 3, 1)
        self.CKSAAP_cnn5 = CNN(128, 256, 3, 1)
        self.h_r1 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.h_r2 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.GAP = nn.AvgPool2d(kernel_size=(2, 2))

        self.CKSAAP_fcnn = FCNN(256, 0.0)

    def forward(self, inputs_p, inputs_n):
        inputs = []
        inputs.extend(inputs_p)
        inputs.extend(inputs_n)
        # print(inputs)
        n = len(inputs)
        m_l = len(inputs[0])

        x_CKSAAP = np.zeros((n, 2400))
        # maxabs = preprocessing.MinMaxScaler()
        for i in range(n):
            x_CKSAAP[i] = CKSAAP(inputs[i])

        cor_feature = torch.tensor(x_CKSAAP, dtype=torch.float32, requires_grad=True)

        batch_size = cor_feature.shape[0]

        x_CNN_in = cor_feature.reshape(batch_size, 6, 20, 20)

        x_o1 = self.CKSAAP_cnn1(x_CNN_in)
        x_o1 = self.CKSAAP_cnn2(x_o1)
        x_o1 = self.mp(x_o1)
        x_o1 = self.dropout(x_o1)

        x_o2 = self.CKSAAP_cnn3(x_o1)
        # x_r = x_CNN_in
        # a = self.h_r1(x_o1)
        x_o2 = x_o2 + self.h_r1(x_o1)
        x_o2 = self.CKSAAP_cnn4(x_o2)
        x_o2 = x_o2 + self.h_r2(x_o1)
        # x_o2 = self.dropout(x_o2)
        x_o = self.CKSAAP_cnn5(x_o2)
        x_o = self.dropout(x_o)

        x_o = self.GAP(x_o)
        x_f = x_o.reshape(batch_size, -1)

        # print(flatten.size())
        outputs = self.CKSAAP_fcnn(x_f)

        return outputs


class EAAC_sub(nn.Module):
    def __init__(self, inputsize):
        super(EAAC_sub, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.EAAC_gru = BGRU(inputsize, 64, 5)
        self.EAAC_cnn1 = CNN(1, 6, 3, 1)
        self.EAAC_cnn2 = CNN(6, 16, 3, 1)
        self.EAAC_cnn3 = CNN(16, 6, 5, 1)
        self.EAAC_cnn4 = CNN(6, 1, 5, 1)
        self.EAAC_att = New_Attention(124, 32)
        self.EAAC_fcnn = FCNN(23 * 124, 0.0)

    def forward(self, inputs_p, inputs_n):
        inputs = []
        inputs.extend(inputs_p)
        inputs.extend(inputs_n)
        # print(inputs)
        n = len(inputs)
        m_l = len(inputs[0])

        x_EAAC = np.zeros((n, 540))
        maxabs = preprocessing.MinMaxScaler()
        for i in range(n):
            x_EAAC[i] = EAAC(inputs[i])

        cor_feature = torch.tensor(x_EAAC, dtype=torch.float32, requires_grad=True)

        batch_size = cor_feature.shape[0]
        cor_feature = cor_feature.reshape(batch_size, 27, 20)
        x_o = self.EAAC_gru(cor_feature)  # [seq,batch_size,hidden_size * 2]
        x_o = self.dropout(x_o)
        x_CNN_in = x_o.permute(1, 0, 2)

        x_o = self.EAAC_cnn1(x_CNN_in.unsqueeze(dim=1))
        x_o = self.EAAC_cnn2(x_o)
        x_o = self.EAAC_cnn3(x_o)
        x_o = self.EAAC_cnn4(x_o)

        x_o = self.EAAC_att(x_o.squeeze(dim=1))
        x_o = self.dropout(x_o)
        x_f = x_o.reshape(batch_size, -1)
        # print(flatten.size())
        outputs = self.EAAC_fcnn(x_f)

        return outputs


class Linear_reg(nn.Module):
    def __init__(self):
        super(Linear_reg, self).__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.w = torch.nn.Parameter(torch.ones(1, 5), requires_grad=True)

    def forward(self, x):
        w_up = self.softmax(self.w)
        out = w_up * x
        out = torch.sum(out, dim=0, keepdim=False)

        return out


def make_data():
    with open("./feature/train_p.txt", "r") as f:
        train_p_string = f.read().split("\n")
    train_p_string = np.array(train_p_string)
    with open("./feature/train_n.txt", "r") as f:
        train_n_string = f.read().split("\n")
    train_n_string = np.array(train_n_string)
    target_p = []
    target_n = []
    for i in range(len(train_n_string)):
        target_n.append([0])
    for i in range(len(train_p_string)):
        target_p.append([1])

    # test_pos_edges = test_pos_edges.T
    # test_neg_edges = test_neg_edges.T
    # test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))
    # test_true_label = np.array(test_true_label, dtype='float32')
    # test_edges = np.vstack((test_pos_edges, test_neg_edges))

    return train_p_string, train_n_string, torch.FloatTensor(target_p), torch.FloatTensor(target_n)


def validate_data():
    with open("./feature/validation_p.txt", "r") as f:
        validate_p_string = f.read().split("\n")
    validate_p_string = np.array(validate_p_string)
    with open("./feature/validation_n.txt", "r") as f:
        validate_n_string = f.read().split("\n")
    validate_n_string = np.array(validate_n_string)
    label_p = []
    label_n = []
    for i in range(len(validate_n_string)):
        label_n.append([0])
    for i in range(len(validate_p_string)):
        label_p.append([1])
    return validate_p_string, validate_n_string, torch.FloatTensor(label_p), torch.FloatTensor(label_n)


def test_data():
    with open("./feature/test_p.txt", "r") as f:
        test_p_string = f.read().split("\n")
    test_p_string = np.array(test_p_string)
    with open("./feature/test_n.txt", "r") as f:
        test_n_string = f.read().split("\n")
    test_n_string = np.array(test_n_string)
    label_p = []
    label_n = []
    for i in range(len(test_n_string)):
        label_n.append([0])
    for i in range(len(test_p_string)):
        label_p.append([1])
    return test_p_string, test_n_string, torch.FloatTensor(label_p), torch.FloatTensor(label_n)


class TestDataSet(torch.utils.data.Dataset):
    def __init__(self, inputs_p, inputs_n, target_p, target_n):
        super(TestDataSet, self).__init__()
        self.inputs_p = inputs_p
        self.inputs_n = inputs_n
        self.target_p = target_p
        self.target_n = target_n

    def __len__(self):
        return self.inputs_p.shape[0]

    def __getitem__(self, idx):
        return self.inputs_p[idx], self.inputs_n[idx], self.target_p[idx], self.target_n[idx]


class validateDataSet(torch.utils.data.Dataset):
    def __init__(self, inputs_p, inputs_n, target_p, target_n):
        super(validateDataSet, self).__init__()
        self.inputs_p = inputs_p
        self.inputs_n = inputs_n
        self.target_p = target_p
        self.target_n = target_n

    def __len__(self):
        return self.inputs_p.shape[0]

    def __getitem__(self, idx):
        return self.inputs_p[idx], self.inputs_n[idx], self.target_p[idx], self.target_n[idx]


class trainDataSet(torch.utils.data.Dataset):
    def __init__(self, inputs_p, inputs_n, target_p, target_n):
        super(trainDataSet, self).__init__()
        self.inputs_p = inputs_p
        self.inputs_n = inputs_n
        self.target_p = target_p
        self.target_n = target_n

    def __len__(self):
        return self.inputs_p.shape[0]

    def __getitem__(self, idx):
        return self.inputs_p[idx], self.inputs_n[idx], self.target_p[idx], self.target_n[idx]


def caculate_metrics(pre_score, real_score):
    fpr, tpr, thresholds = metrics.roc_curve(real_score, pre_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    y_score = [0 if j < 0.5 else 1 for j in pre_score]

    acc = accuracy_score(real_score, y_score)
    confusion = confusion_matrix(real_score, y_score)
    TN, FP, FN, TP = confusion.ravel()
    sensitivity = recall_score(real_score, y_score)
    specificity = TN / float(TN + FP)
    MCC = matthews_corrcoef(real_score, y_score)
    PR = metrics.precision_score(real_score, y_score)

    print(f'sn:{sensitivity},sp:{specificity},mcc:{MCC},ACC:{acc},auc:{auc},PR:{PR}')

    metric_result = [auc, acc, sensitivity, specificity, MCC, PR]

    return metric_result


def train(sub, model, e):
    criterion = nn.BCELoss()
    optimizer = optim.NAdam(model.parameters(), lr=0.001, weight_decay=0.001)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))

    best_acc = 0.0
    best_mcc = -1.0
    for epoch in (range(e + 1)):
        model.train()
        print("epoch", epoch + 1)
        epo_score, epo_label = [], []
        for inputs_p, inputs_n, target_p, target_n in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs_p, inputs_n)
            target_p = np.array(target_p)
            target_n = np.array(target_n)
            target_p = torch.tensor(target_p, dtype=torch.float32)
            target_n = torch.tensor(target_n, dtype=torch.float32)
            target = torch.cat((target_p, target_n), 0)
            # print(target)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # print("loss={loss:.3f};",loss.item())###
            batch_score = outputs.cpu().detach().numpy()
            epo_score = np.append(epo_score, batch_score)
            epo_label = np.append(epo_label, target.numpy())
        train_metric = caculate_metrics(epo_score, epo_label)  # ***

        val_result = evalute(model, validate_loader)
        if val_result[4] >= best_mcc and val_result[1] >= best_acc:
            best_epoch = epoch
            best_mcc = val_result[4]
            best_acc = val_result[1]
            torch.save(model.state_dict(), './new/model_{}.mdl'.format(sub))
        print('best acc：', best_acc, 'best epoch：', best_epoch)


def test(model):
    epo_score = []
    epo_label = []
    all_score = []
    label = []
    result = []
    label_p = []
    result_p = []
    label_n = []
    result_n = []
    # for i in range(5):
    #
    #     model1.load_state_dict(torch.load('./model/best_{}.mdl'.format(i)))
    model.eval()
    with torch.no_grad():
        for inputs_p, inputs_n, target_p, target_n in test_loader:
            outputs = model(inputs_p, inputs_n)

            target_p = np.array(target_p)
            target_n = np.array(target_n)
            target_p = torch.tensor(target_p, dtype=torch.float32)
            target_n = torch.tensor(target_n, dtype=torch.float32)
            target = torch.cat((target_p, target_n), 0)
            # outputs_z = torch.split(outputs,int(len(outputs)/2), dim=0)
            # outputs_p = outputs_z[0]
            # outputs_n = outputs_z[1]
            # outputs_p = outputs_p.detach().numpy()
            # outputs_n = outputs_n.detach().numpy()
            batch_score = outputs.cpu().detach().numpy()
            epo_score = np.append(epo_score, batch_score)
            epo_label = np.append(epo_label, target.numpy())
    # all_score = np.append(all_score,epo_score)
    #
    # final_score = all_score[0]*0.13+all_score[1]*0.19+all_score[2]*0.28+all_score[3]*0.27+all_score[4]*0.13
    metric = caculate_metrics(epo_score, epo_label)
    # torch.save(model.state_dict(), './model/test_2.mdl')
    return epo_score, epo_label


def evalute(model, validater):
    epo_score = []
    epo_label = []
    model.eval()
    with torch.no_grad():
        # criterion = nn.BCELoss()
        for validate_p, validate_n, target_p, target_n in validater:
            outputs = model(validate_p, validate_n)
            target_p = np.array(target_p)
            target_n = np.array(target_n)
            target_p = torch.tensor(target_p, dtype=torch.float32)
            target_n = torch.tensor(target_n, dtype=torch.float32)
            target = torch.cat((target_p, target_n), 0)
            # loss = criterion(outputs, target)
            # print('loss =', '{:.6f}'.format(loss))
            batch_score = outputs.cpu().detach().numpy()
            epo_score = np.append(epo_score, batch_score)
            epo_label = np.append(epo_label, target.numpy())

        valid_metric = caculate_metrics(epo_score, epo_label)

    return valid_metric


def stack_en(model, validater):
    epo_score = []
    epo_label = []
    model.eval()
    with torch.no_grad():
        criterion = nn.BCELoss()
        for validate_p, validate_n, target_p, target_n in validater:
            outputs = model(validate_p, validate_n)
            target_p = np.array(target_p)
            target_n = np.array(target_n)
            target_p = torch.tensor(target_p, dtype=torch.float32)
            target_n = torch.tensor(target_n, dtype=torch.float32)
            target = torch.cat((target_p, target_n), 0)
            loss = criterion(outputs, target)
            # print('loss =', '{:.6f}'.format(loss))
            batch_score = outputs.cpu().detach().numpy()
            epo_score = np.append(epo_score, batch_score)
            epo_label = np.append(epo_label, target.numpy())

        valid_metric = caculate_metrics(epo_score, epo_label)

    return valid_metric


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # setup_seed(1234)

    batch_size = 32
    hidden_size = 128
    d_k = d_v = 32  # dimension of K(=Q), V
    n_layers = 1  # number of Encoder of Decoder Layer
    n_heads = 4  # number of heads in Multi-Head Attention
    initial_lr = 0.001
    # model = BE_sub(54)
    inputs_p, inputs_n, target_p, target_n = make_data()
    validate_p, validate_n, labelv_p, labelv_n = validate_data()
    test_p, test_n, label_p, label_n = test_data()
    train_loader = torch.utils.data.DataLoader(trainDataSet(inputs_p, inputs_n, target_p, target_n), batch_size,
                                               shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validateDataSet(validate_p, validate_n, labelv_p, labelv_n),
                                                  batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(TestDataSet(test_p, test_n, label_p, label_n), batch_size, shuffle=False)
    # for e in range(0,30):
    e = 50
    model1 = BE_sub(20)
    print("-----train1-----")
    train(0, model1, e)
    model2 = Blosum62_sub(20)
    print("-----train2-----")
    train(1, model2, e)
    model3 = AAI_sub(531)
    print("-----train3-----")
    train(2, model3, e)
    model4 = CKSAAP_sub(6)
    print("-----train4-----")
    train(3, model4, e)
    model5 = EAAC_sub(20)
    print("-----train5-----")
    train(4, model5, e)
    # torch.save(model1.state_dict(), './new/model_1.mdl')
    # torch.save(model2.state_dict(), './new/model_2.mdl')
    # torch.save(model3.state_dict(), './new/model_3.mdl')
    # torch.save(model4.state_dict(), './new/model_4.mdl')
    # torch.save(model5.state_dict(), './new/model_5.mdl')
    # model.load_state_dict(torch.load("models/best.mdl"))
    model1.load_state_dict(torch.load("./new/model_0.mdl"))
    model2.load_state_dict(torch.load("./new/model_1.mdl"))
    model3.load_state_dict(torch.load("./new/model_2.mdl"))
    model4.load_state_dict(torch.load("./new/model_3.mdl"))
    model5.load_state_dict(torch.load("./new/model_4.mdl"))

    print("-----test-----")
    model1_s, model1_l = test(model1)
    model2_s, model2_l = test(model2)
    model3_s, model3_l = test(model3)
    model4_s, model4_l = test(model4)
    model5_s, model5_l = test(model5)

    final_score = (model1_s * 0.13) + (model2_s * 0.19) + (model3_s * 0.28) + (model4_s * 0.27) + (model5_s * 0.13)
    test_metric = caculate_metrics(final_score, model5_l)
