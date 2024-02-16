import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import KFold
from model import BE_sub, Blosum62_sub, AAI_sub, CKSAAP_sub, EAAC_sub
import torch.utils.data.dataloader as Dataloader
from datapro import SeqDataset


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

    metric_result = [auc, acc, sensitivity, specificity, MCC, PR]

    return metric_result


def train(sub, model, trainloader, validloader, epoch, lear,w_d):
    criterion = nn.BCELoss()
    optimizer = optim.NAdam(model.parameters(), lr=lear, weight_decay=w_d)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))

    best_acc = 0
    best_mcc = -1.0
    # best_auc = 0.0
    for epo in range(epoch):
        model.train()
        print("epoch", epo + 1)
        epo_score, epo_label = [], []
        for i, item in enumerate(trainloader):
            data, label = item
            traindata = data
            trainlabel = label.cuda()
            outputs = model(traindata)

            loss = criterion(outputs, trainlabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            # print("loss={loss:.3f};",loss.item())###
            batch_score = outputs.cpu().detach().numpy()
            epo_score = np.append(epo_score, batch_score)
            epo_label = np.append(epo_label, label.numpy())
        print("loss:", loss.item())
        train_metric = caculate_metrics(epo_score, epo_label)

        val_score, val_label = evalute(model, validloader)
        valid_metric = caculate_metrics(val_score, val_label)

        if valid_metric[4] >= best_mcc and valid_metric[1] >= best_acc:
            best_epoch = epo + 1
            best_mcc = valid_metric[4]
            best_auc = valid_metric[0]
            best_acc = valid_metric[1]
            torch.save(model.state_dict(), './one-test/test_sub{}.mdl'.format(sub))
        print('best acc：', best_acc, 'best epoch：', best_epoch)


def test(model, testloader):
    epo_score = []
    epo_label = []

    model.eval()
    with torch.no_grad():
        for i, item in enumerate(testloader):
            data, label = item
            traindata = data
            # trainlabel = label.cuda()
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
        for i, item in enumerate(validater):
            data, label = item
            traindata = data
            trainlabel = label.cuda()
            outputs = model(traindata)
            loss = criterion(outputs, trainlabel)
            # print('loss =', '{:.6f}'.format(loss))
            batch_score = outputs.cpu().detach().numpy()
            epo_score = np.append(epo_score, batch_score)
            epo_label = np.append(epo_label, label.numpy())

        # valid_metric = caculate_metrics(epo_score, epo_label)

    return epo_score, epo_label


def train_test(train_data, epoch, batch_size):

    train_new_seqs = train_data['train_seq']
    train_new_labels = train_data['train_label']

    valid_seqs = train_data['valid_seq']
    valid_labels = train_data['valid_label']

    test_seqs = train_data['test_seq']
    test_labels = train_data['test_label']

    torch.manual_seed(3701)


    trainData = SeqDataset(train_new_seqs, train_new_labels)
    validData = SeqDataset(valid_seqs, valid_labels)
    testData = SeqDataset(test_seqs, test_labels)

    trainloader = Dataloader.DataLoader(trainData, batch_size, shuffle=True)
    validloader = Dataloader.DataLoader(validData, batch_size, shuffle=False)
    testloader = Dataloader.DataLoader(testData, batch_size, shuffle=False)

    # The five models are trained separately. Then load them for integration.

    print("-----train1-----")
    model1 = BE_sub(20,256,2,0.1,0.0)
    model1.cuda()
    # train(0, model1, trainloader, validloader,100,0.00003,0.0001)


    print("-----train2-----")
    model2 = Blosum62_sub(20,128,3,0.2,0.0)
    model2.cuda()
    # train(1, model2, trainloader, validloader,100,0.00005,0.0001)

    print("-----train3-----")
    model3 = AAI_sub(531,256,4,0.1,0.1)
    model3.cuda()
    # train(2, model3, trainloader, validloader,50,0.00005,0.0001)

    print("-----train4-----")
    model4 = CKSAAP_sub(6,64,2,0.6,0.1)
    model4.cuda()
    # train(3, model4, trainloader, validloader,100,0.002,0.0001)#

    print("-----train5-----")
    model5 = EAAC_sub(20,128,2,0.1,0.1)
    model5.cuda()
    # train(4, model5, trainloader, validloader,150,0.00005,0.0001)


    model1.load_state_dict(torch.load("./one-test/test_sub0.mdl"))
    model2.load_state_dict(torch.load("./one-test/test_sub1.mdl"))
    model3.load_state_dict(torch.load("./one-test/test_sub2.mdl"))
    model4.load_state_dict(torch.load("./one-test/test_sub3.mdl"))
    model5.load_state_dict(torch.load("./one-test/test_sub4.mdl"))

    print("-----valid-----")
    val_score1, val_label1 = evalute(model1, validloader)
    val_score2, val_label2 = evalute(model2, validloader)
    val_score3, val_label3 = evalute(model3, validloader)
    val_score4, val_label4 = evalute(model4, validloader)
    val_score5, val_label5 = evalute(model5, validloader)

    all_val_pre = np.vstack((val_score1, val_score2, val_score3, val_score4, val_score5))

    all_val = torch.from_numpy(all_val_pre)
    val_label = torch.from_numpy(valid_labels)
    # initial_w = torch.mean(all_val,axis=1)
    # w_v = stack_en(all_val,val_label,all_val_acc)
    w_v = stack_en(all_val, val_label)
    #
    print("-----test-----")
    model1_s, model1_l = test(model1, testloader)
    model2_s, model2_l = test(model2, testloader)
    model3_s, model3_l = test(model3, testloader)
    model4_s, model4_l = test(model4, testloader)
    model5_s, model5_l = test(model5, testloader)

    print("-----final-test-----")
    all_test_pre = np.vstack((model1_s, model2_s, model3_s, model4_s, model5_s))

    score = all_test_pre * w_v
    final_score = np.sum(score, axis=0, keepdims=False)

    final_metric = caculate_metrics(final_score, test_labels)

    ##looking...Integration with appropriate weights.
    fix_score = (model1_s * 0.35) + (model2_s * 0.23) + (model3_s * 0.13) + (model4_s * 0.13) + (model5_s * 0.16)
    # fix_score = (model1_s * 0.349) + (model2_s * 0.235) + (model3_s * 0.133) + (model4_s * 0.128) + (model5_s * 0.155)

    fix_metric = caculate_metrics(fix_score, test_labels)

    # gc.collect()
    torch.cuda.empty_cache()

    return fix_metric


class Linear_reg(nn.Module):
    def __init__(self):
        super(Linear_reg, self).__init__()
        torch.manual_seed()
        self.softmax = nn.Softmax(dim=-1)
        # self.w = torch.nn.Parameter(torch.tensor(w),requires_grad=True)
        self.w = torch.nn.Parameter(torch.randn(1, 5), requires_grad=True)

    def forward(self, x):
        w_up = self.softmax(self.w)
        w_up = w_up.t()

        out = (w_up * x)
        out = torch.sum(out, dim=0, keepdim=False)

        return out.float(), w_up


def stack_en(x, label):
    modelE = Linear_reg()
    # modelE.load_state_dict(torch.load("./ES_model/Init_Weight.mdl"))
    # modelE.cuda()
    optimizer = optim.NAdam(modelE.parameters(), lr=0.001, weight_decay=0.0001)

    epo_score = []
    epo_label = []
    modelE.cuda()
    criterion = nn.MSELoss()

    for e in range(300):
        modelE.train()
        outputs, w_value = modelE(x.cuda())
        loss = criterion(outputs, label.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('loss =', '{:.6f}'.format(loss))
        batch_score = outputs.cpu().detach().numpy()
        epo_score = np.append(epo_score, batch_score)
        epo_label = np.append(epo_label, label.numpy())

        es_metric = caculate_metrics(epo_score, epo_label)

        best_w = w_value.cpu().detach().numpy()
        print('w', best_w)

    return best_w
