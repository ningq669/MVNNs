import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import tqdm
import random
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold
from model import BE_sub,Blosum62_sub,AAI_sub,CKSAAP_sub,EAAC_sub
import torch.utils.data.dataloader as Dataloader
from datapro import SeqDataset,make_alldata





def caculate_metrics(pre_score, real_score):
    fpr, tpr, thresholds = metrics.roc_curve(real_score, pre_score, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    y_score = [0 if j < 0.55 else 1 for j in pre_score]

    acc = metrics.accuracy_score(real_score, y_score)
    confusion = metrics.confusion_matrix(real_score, y_score)
    TN, FP, FN, TP = confusion.ravel()
    sensitivity = metrics.recall_score(real_score, y_score)
    specificity = TN / float(TN + FP)
    MCC = metrics.matthews_corrcoef(real_score, y_score)
    PR = metrics.precision_score(real_score, y_score)
    print(f'sn:{sensitivity},sp:{specificity},mcc:{MCC},ACC:{acc},auc:{auc},PR:{PR}')

    metric_result = [auc, acc, sensitivity, specificity, MCC,PR]

    return metric_result


def train(sub, model,trainloader,validloader,epoch,lear,w_d,k):
    criterion = nn.BCELoss()
    optimizer = optim.NAdam(model.parameters(), lr=lear, weight_decay=w_d)

    for epo in range(epoch):
        model.train()
        # if epo%50==0:
        print("epoch", epo + 1)
        epo_score, epo_label = [], []
        for i,item in enumerate(trainloader):
            data,label = item
            traindata = data
            trainlabel = label.cuda()
            outputs = model(traindata)

            loss = criterion(outputs, trainlabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_score = outputs.cpu().detach().numpy()
            epo_score = np.append(epo_score, batch_score)
            epo_label = np.append(epo_label, label.numpy())
        print("loss:", loss.item())
        train_metric = caculate_metrics(epo_score, epo_label)




def test(model,testloader):
    epo_score = []
    epo_label = []
    all_score = []
    label = []
    result = []
    label_p = []
    result_p = []
    label_n = []
    result_n = []

    model.eval()
    with torch.no_grad():
        for i,item in enumerate(testloader):
            data,label = item
            traindata = data
            outputs = model(traindata)

            batch_score = outputs.cpu().detach().numpy()
            epo_score = np.append(epo_score, batch_score)
            epo_label = np.append(epo_label, label.numpy())

    return epo_score, epo_label


def evalute(model, validater):
    epo_score = []
    epo_label = []
    model.eval()
    with torch.no_grad():
        criterion = nn.BCELoss()
        for i,item in enumerate(validater):
            data, label = item
            traindata = data
            trainlabel = label.cuda()
            outputs = model(traindata)
            loss = criterion(outputs, trainlabel)
            # print('loss =', '{:.6f}'.format(loss))
            batch_score = outputs.cpu().detach().numpy()
            epo_score = np.append(epo_score, batch_score)
            epo_label = np.append(epo_label, label.numpy())

    return epo_score, epo_label


def train_test(train_data,batch_size):


    train_all_seqs = train_data['train_all_seq']
    train_all_labels = train_data['train_label']

    kfolds = 10
    torch.manual_seed(3701)



    kf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=8888)
    train_idx,test_idx=[],[]

    bestResult_fix1 = []
    bestResult_fix2 = []



    for train_index,test_index in kf.split(train_all_seqs,train_all_labels):
        train_idx.append(train_index)
        test_idx.append(test_index)

    for k in range(10):
        print("fold",k+1)
        train_seqs,train_labels =train_all_seqs[train_idx[k]],train_all_labels[train_idx[k]]
        test_seqs,test_labels = train_all_seqs[test_idx[k]],train_all_labels[test_idx[k]]
        trainData = SeqDataset(train_seqs, train_labels)
        testData = SeqDataset(test_seqs, test_labels)

        trainloader = Dataloader.DataLoader(trainData, batch_size, shuffle=True)##因为trainloader会shuffle？
        testloader = Dataloader.DataLoader(testData, batch_size, shuffle=False)
        #
        print("-----train1-----")
        model1 = BE_sub(20,256, 2, 0.1, 0.0)
        model1.cuda()
        # train(0, model1, trainloader, testloader, 100, 0.00003, 0.0001, k)


        print("-----train2-----")
        model2 = Blosum62_sub(20, 128, 2, 0.3, 0.1)
        model2.cuda()
        # train(1, model2, trainloader, testloader, 100, 0.00005, 0.0001, k)

        print("-----train3-----")
        model3 = AAI_sub(531, 64, 4, 0.2, 0.1)
        model3.cuda()
        # train(2, model3, trainloader, testloader, 50, 0.00005, 0.0001, k)


        print("-----train4-----")
        model4 = CKSAAP_sub(6,128, 2, 0.2, 0.0)
        model4.cuda()
        # train(3, model4, trainloader, testloader,100, 0.0002, 0.0001, k)#



        print("-----train5-----")
        model5 = EAAC_sub(20,256,2,0.1,0.0)
        model5.cuda()
        # train(4, model5, trainloader, testloader,100,0.00003,0.0001,k)


        #### model.load_state_dict(torch.load("models/best.mdl"))
        model1.load_state_dict(torch.load("./cross-valid/fold{}_sub0.mdl".format(k)))
        model2.load_state_dict(torch.load("./cross-valid/fold{}_sub1.mdl".format(k)))
        model3.load_state_dict(torch.load("./cross-valid/fold{}_sub2.mdl".format(k)))
        model4.load_state_dict(torch.load("./cross-valid/fold{}_sub3.mdl".format(k)))
        model5.load_state_dict(torch.load("./cross-valid/fold{}_sub4.mdl".format(k)))

        print("-----test-----")
        model1_s, model1_l = test(model1, testloader)
        model2_s, model2_l = test(model2, testloader)
        model3_s, model3_l = test(model3, testloader)
        model4_s, model4_l = test(model4, testloader)
        model5_s, model5_l = test(model5, testloader)
        metric1 = caculate_metrics(model1_s, model1_l)
        metric2 = caculate_metrics(model2_s, model2_l)
        metric3 = caculate_metrics(model3_s, model3_l)
        metric4 = caculate_metrics(model4_s, model4_l)
        metric5 = caculate_metrics(model5_s, model5_l)
        print(metric1)
        print(metric2)
        print(metric3)
        print(metric4)
        print(metric5)


        print("-----final-test-----")
        fix_score = (model1_s * 0.35) + (model2_s * 0.23) + (model3_s * 0.13) + (model4_s * 0.13) + (model5_s * 0.16)

        fix_metric = caculate_metrics(fix_score, test_labels)

        bestResult_fix1.append(fix_metric)

    print(np.array(bestResult_fix1))
    cv_fix1_me = np.mean(np.array(bestResult_fix1),axis=0)
    print(cv_fix1_me)

    return cv_fix1_me



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
    datasets = make_alldata()
    epoch = 30
    result = train_test(datasets, batch_size)