'''
#edit_time: 2018-2-19
#editor: L
#本文件为神经网络模型文件，内容包含了模型参数的设定，抽取训练集和测试集
#训练并测试模型并最终将模型保存为GRU.pkl
数据来源为父目录中dataset层级下final_data文件夹
'''
import os
import glob
import pandas as pd
import torch 
import random
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
#学习效率
learning_rate = 0.005
#特征个数
in_dim = 20
#线性层隐藏层神经元个数
hidden_dim = 64
#GRU隐藏层层数
n_layers = 2
#分类的类别个数
n_class = 5
#训练次数
times = 5
#训练集和测试集选取密度（数值越大，密度越小）
k = 2

#构建神经网络，第一层为线性层，第二层为GRU曾，第三层为dropout层，用于避免过拟合
#第四层为线性层，最后一层为线性分类层
class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layers, n_class):
        super(Rnn, self).__init__()
        self.hidden_dim = hidden_dim
        self.ready = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, 2 * hidden_dim, n_layers, batch_first=True)
        self.outer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_class)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.ready(x)
        out, _ = self.gru(out)
        out = self.outer(out)
        out = self.classifier(out)
        out = self.drop(out)
        return out
#把参数传入神经网络
model = Rnn(in_dim, hidden_dim, n_layers, n_class)

model = model.cuda()
# 定义loss和optimizer，使用交叉熵衡量模型损失，使用SGD优化器优化网络内部参数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
#构建训练集和测试集，训练集10%，测试集20%
FILE_LST = glob.glob('../dataset/final_data/*.csv')
fnb = int(0.1 * len(FILE_LST))
SL_LST = random.sample(FILE_LST, 3 * fnb)
random.shuffle(SL_LST)
TRAIN_LST = SL_LST[:fnb]
TEST_LST = SL_LST[fnb:]
#进行训练
for i in range(times):
    for train_file in TRAIN_LST:    
        df = pd.read_csv(train_file)
        df = df[:-48]
        df = df.set_index(['date', 'seqNo'])
        arr = np.array(df)
        data = arr[:,:-1]
        label = arr[:,-1]
        data = Variable(torch.from_numpy(data).float().view(-1,len(df),in_dim)).cuda()
        label = Variable(torch.from_numpy(label).long()).cuda()
        out = model(data)
        loss = criterion(out[0], label)
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()
        print(loss)
#进行测试，计算测试集平均五分类正确率，三分类正确率（涨，平，跌）和平均损失
print('***********************testing******************')
accr_rate_lst = []
biaccr_rate_lst = []
lss_lst = []
for test_file in TEST_LST:    
    df = pd.read_csv(test_file)
    df = df[:-48]
    df = df.set_index(['date', 'seqNo'])
    arr = np.array(df)
    data = arr[:,:-1]
    label = arr[:,-1]
    data = Variable(torch.from_numpy(data).float().view(-1,len(df),in_dim)).cuda()
    label = Variable(torch.from_numpy(label).long()).cuda()
    out = model(data)
    loss = criterion(out[0], label)
    prediction = torch.max(F.softmax(out[0],dim=1), 1)[1].cpu()
    prediction = prediction.data.numpy()
    prediction = pd.Series(prediction)
    accr = df['labels'].reset_index(drop=True) - prediction
    accr =  accr.map(lambda x: 1 if x == 0 else 0)
    bipre = prediction.map(lambda x: 0 if x < 2 else 1 if x == 2 else 2 )
    bilb = df['labels'].reset_index(drop=True)
    bilb = bilb.map(lambda x: 0 if x < 2 else 1 if x == 2 else 2 )
    biaccr = bilb - bipre
    biaccr = biaccr.map(lambda x: 1 if x == 0 else 0)
    biaccr_rate = biaccr.sum()/len(biaccr)
    biaccr_rate_lst.append(biaccr_rate)
    accr_rate = accr.sum()/len(accr)
    accr_rate_lst.append(accr_rate)
    loss = loss.cpu()
    lss_lst.append(loss.data.numpy()[0])
    print(accr_rate)
    print(biaccr_rate)
print('测试集平均准确率为：')
print(sum(accr_rate_lst)/len(accr_rate_lst))
print('误差在1以内的平均准确率为：')
print(sum(biaccr_rate_lst)/len(biaccr_rate_lst))
print('平均损失：')
print(sum(lss_lst)/len(lss_lst))
#保存模型
torch.save(model, 'GRU.pkl')
