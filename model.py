import torch
import torch.nn as nn
import numpy as np
# import torch.optim as optim
from sklearn import preprocessing
# from torch.optim.lr_scheduler import LambdaLR
from collections import Counter
import re


class New_Attention(nn.Module):
    def __init__(self, inputsize, hidden_size):
        super(New_Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=inputsize, num_heads=4, dropout=0.3, batch_first=True)
        self.wq = nn.Linear(inputsize, hidden_size, bias=False)
        self.wk = nn.Linear(inputsize, hidden_size, bias=False)
        self.wv = nn.Linear(inputsize, hidden_size, bias=False)

    def forward(self, x):
        # Q = self.wq(x)
        # K = self.wk(x)
        # V = self.wv(x)
        attn_output, attn_output_weights = self.attention(x, x, x)

        return attn_output


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.dk = 32

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.dk)  # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiHeadAttention, self).__init__()
        self.n_head = 4
        self.dk = 32
        self.dim = d_model
        self.l_norm = nn.LayerNorm(self.dim)
        self.W_Q = nn.Linear(d_model, self.dk * self.n_head, bias=False)
        self.W_K = nn.Linear(d_model, self.dk * self.n_head, bias=False)
        self.W_V = nn.Linear(d_model, self.dk * self.n_head, bias=False)
        self.fc = nn.Linear(self.n_head * self.dk, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size1 = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size1, -1, self.n_head, self.dk).transpose(1,
                                                                                    2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size1, -1, self.n_head, self.dk).transpose(1,
                                                                                    2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size1, -1, self.n_head, self.dk).transpose(1,
                                                                                    2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size1, -1,
                                                  self.n_head * self.dk)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        f_o = self.l_norm(output + residual)
        return f_o


class AtentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AtentionLayer, self).__init__()
        self.dim = d_model
        self.enc_self_attn = MultiHeadAttention(self.dim)

    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.n_layer = 2
        self.dim = d_model
        self.layers = nn.ModuleList([AtentionLayer(self.dim) for _ in range(self.n_layer)])

    def forward(self, enc_inputs):
        for layer in self.layers:
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
        # batch_size_X = X.shape[0]
        input_X = X.transpose(0, 1)

        outputs_lstm, h_n = self.lstm(input_X)

        return outputs_lstm


class BGRU(nn.Module):
    def __init__(self, input_size_gru, hidden_size, layer_num):
        super(BGRU, self).__init__()

        self.gru = nn.GRU(input_size=input_size_gru, hidden_size=hidden_size, num_layers=layer_num, bidirectional=True,
                          batch_first=True)

    def forward(self, X):
        input_X = X.transpose(0, 1)

        outputs_gru, h_n = self.gru(input_X)

        return outputs_gru


class FCNN(nn.Module):
    def __init__(self, input_sizes, drop):
        super(FCNN, self).__init__()

        self.dropout = nn.Dropout(p=drop)
        self.fc1 = nn.Linear(input_sizes, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        outputs = self.sigmoid(x)

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
    AAindex = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)

    AA = 'ARNDCQEGHILKMFPSTWYV'
    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    sequence = gene.replace("*", "-")
    code = []
    for aa in sequence:
        if aa == '-':
            for j in AAindex:
                code.append(0)
            continue
        for j in AAindex:
            code.append(j[index[aa]])
    final_code = np.array(code)
    final_c = np.reshape(final_code, (31, 531))
    return final_c


def CKSAAP(gene, gap=5):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    sequence = gene.replace("*", "-")
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
    def __init__(self, inputsize,hin_size,hid_layer,hid_drop,fc_drop):
        super(BE_sub, self).__init__()

        self.BE_gru = BGRU(inputsize, hin_size, hid_layer)
        self.dropout = nn.Dropout(p=hid_drop)
        self.BE_fcnn = FCNN(31 *hin_size*2, fc_drop)

    def forward(self, inputs):
        n = len(inputs)
        m_l = len(inputs[0])
        x_BE = np.zeros((n, m_l, 20))

        maxabs = preprocessing.MinMaxScaler()
        for i in range(n):
            x_BE[i] = BE(inputs[i])
            x_BE[i] = maxabs.fit_transform(x_BE[i])
        cor_feature = torch.tensor(x_BE, dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)

        batch_size = cor_feature.shape[0]
        x_o = self.BE_gru(cor_feature)  # [seq,batch_size,hidden_size * 2]
        x_o = x_o.permute(1, 0, 2)
        x_o = self.dropout(x_o)

        x_f = x_o.reshape(batch_size, -1)
        outputs = self.BE_fcnn(x_f)
        outputs = outputs.squeeze(dim=1)

        return outputs


class Blosum62_sub(nn.Module):
    def __init__(self, inputsize,hin_size,hid_layer,hid_drop,fc_drop):
        super(Blosum62_sub, self).__init__()
        self.dropout = nn.Dropout(p=hid_drop)
        self.Blosum62_lstm = BLSTM(inputsize, hin_size, hid_layer)
        self.Blosum62_att = SelfAttention(hin_size * 2)
        self.Blosum62_fcnn = FCNN(31 * hin_size * 2, fc_drop)

    def forward(self, inputs):
        n = len(inputs)
        m_l = len(inputs[0])

        x_62 = np.zeros((n, m_l, 20))

        maxabs = preprocessing.MinMaxScaler()
        for i in range(n):
            x_62[i] = BLOSUM62(inputs[i])
            x_62[i] = maxabs.fit_transform(x_62[i])

        cor_feature = torch.tensor(x_62, dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)

        batch_size = cor_feature.shape[0]

        x_o = self.Blosum62_lstm(cor_feature)  # [seq,batch_size,hidden_size * 2]
        x_o = self.dropout(x_o)
        x_attention_in = x_o.permute(1, 0, 2)

        x_no = self.Blosum62_att(x_attention_in)
        x_no = self.dropout(x_no)
        x_f = x_no.reshape(batch_size, -1)

        outputs = self.Blosum62_fcnn(x_f)
        outputs = outputs.squeeze(dim=1)

        return outputs


class AAI_sub(nn.Module):
    def __init__(self, inputsize,hin_size,hid_layer,hid_drop,fc_drop):
        super(AAI_sub, self).__init__()
        self.dropout = nn.Dropout(p=hid_drop)
        self.AAI_gru = BGRU(inputsize, hin_size, hid_layer)
        self.AAI_att = SelfAttention(hin_size * 2)
        self.AAI_fcnn = FCNN(31 * hin_size * 2, fc_drop)

    def forward(self, inputs):
        n = len(inputs)
        m_l = len(inputs[0])

        x_AAI = np.zeros((n, m_l, 531))
        maxabs = preprocessing.MinMaxScaler()
        for i in range(n):
            x_AAI[i] = AAI(inputs[i])
            x_AAI[i] = maxabs.fit_transform(x_AAI[i])

        cor_feature = torch.tensor(x_AAI, dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)

        batch_size = cor_feature.shape[0]

        x_o = self.AAI_gru(cor_feature)  # [seq,batch_size,hidden_size * 2]
        x_o = self.dropout(x_o)
        x_attention_in = x_o.permute(1, 0, 2)

        x_no = self.AAI_att(x_attention_in)
        x_no = self.dropout(x_no)

        x_f = x_no.reshape(batch_size, -1)

        outputs = self.AAI_fcnn(x_f)
        outputs = outputs.squeeze(dim=1)

        return outputs

class CKSAAP_sub(nn.Module):
    def __init__(self, inputsize,hin_size,hid_layer,hid_drop,fc_drop):
        super(CKSAAP_sub, self).__init__()
        self.dropout = nn.Dropout(p=hid_drop)
        self.CKSAAP_cnn1 = CNN(inputsize, hin_size, 3, 2)
        self.CKSAAP_cnn2 = CNN(hin_size, hin_size, 3, 2)
        self.mp = nn.MaxPool2d(stride=2, kernel_size=3)
        self.CKSAAP_cnn3 = CNN(hin_size, hin_size, 3, 1)
        self.CKSAAP_cnn4 = CNN(hin_size, (hin_size*2), 3, 1)
        self.CKSAAP_cnn5 = CNN((hin_size*2), (hin_size*4), 3, 1)
        self.h_r1 = nn.Conv2d(hin_size, hin_size, kernel_size=1, stride=1, bias=False)
        self.h_r2 = nn.Conv2d(hin_size, (hin_size*2), kernel_size=1, stride=1, bias=False)
        self.GAP = nn.AvgPool2d(kernel_size=(2, 2))
        self.CKSAAP_fcnn = FCNN((hin_size*4), fc_drop)

    def forward(self, inputs):
        n = len(inputs)
        x_CKSAAP = np.zeros((n, 2400))

        for i in range(n):
            x_CKSAAP[i] = CKSAAP(inputs[i])

        cor_feature = torch.tensor(x_CKSAAP, dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)

        batch_size = cor_feature.shape[0]

        x_CNN_in = cor_feature.reshape(batch_size, 6, 20, 20)

        x_o1 = self.CKSAAP_cnn1(x_CNN_in)
        x_o1 = self.dropout(x_o1)
        x_o1 = self.CKSAAP_cnn2(x_o1)
        x_o1 = self.dropout(x_o1)
        x_o1 = self.mp(x_o1)

        x_o2 = self.CKSAAP_cnn3(x_o1)
        x_o2 = self.dropout(x_o2)

        x_o2 = x_o2 + self.h_r1(x_o1)
        x_o2 = self.CKSAAP_cnn4(x_o2)
        x_o2 = self.dropout(x_o2)
        x_o2 = x_o2 + self.h_r2(x_o1)
        x_o = self.CKSAAP_cnn5(x_o2)
        x_o = self.dropout(x_o)

        x_o = self.GAP(x_o)

        x_f = x_o.reshape(batch_size, -1)

        outputs = self.CKSAAP_fcnn(x_f)
        outputs = outputs.squeeze(dim=1)

        return outputs


class EAAC_sub(nn.Module):
    def __init__(self, inputsize,hin_size,hid_layer,hid_drop,fc_drop):
        super(EAAC_sub, self).__init__()
        self.dropout = nn.Dropout(p=hid_drop)
        self.EAAC_gru = BGRU(inputsize, hin_size, hid_layer)
        self.EAAC_cnn1 = CNN(1, 6, 3, 1)
        self.EAAC_cnn2 = CNN(6, 16, 3, 1)
        self.EAAC_cnn3 = CNN(16, 6, 5, 1)
        self.EAAC_cnn4 = CNN(6, 1, 5, 1)

        self.EAAC_att = SelfAttention((hin_size*2)-4)
        self.EAAC_fcnn = FCNN(23 * ((hin_size*2)-4), fc_drop)

    def forward(self, inputs):
        n = len(inputs)

        x_EAAC = np.zeros((n, 540))

        for i in range(n):
            x_EAAC[i] = EAAC(inputs[i])

        cor_feature = torch.tensor(x_EAAC, dtype=torch.float32, device=torch.device('cuda'), requires_grad=True)

        batch_size = cor_feature.shape[0]
        cor_feature = cor_feature.reshape(batch_size, 27, 20)
        x_o = self.EAAC_gru(cor_feature)  # [seq,batch_size,hidden_size * 2]
        x_o = self.dropout(x_o)
        x_CNN_in = x_o.permute(1, 0, 2)

        x_o = self.EAAC_cnn1(x_CNN_in.unsqueeze(dim=1))
        # x_o = self.dropout(x_o)
        x_o = self.EAAC_cnn2(x_o)
        # x_o = self.dropout(x_o)
        x_o = self.EAAC_cnn3(x_o)
        # x_o = self.dropout(x_o)
        x_o = self.EAAC_cnn4(x_o)
        x_o = self.dropout(x_o)

        x_o = self.EAAC_att(x_o.squeeze(dim=1))

        x_f = x_o.reshape(batch_size, -1)

        outputs = self.EAAC_fcnn(x_f)
        outputs = outputs.squeeze(dim=1)

        return outputs
