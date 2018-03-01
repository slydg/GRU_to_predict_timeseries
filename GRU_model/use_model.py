'''
#edit_time: 2018-2-19
#editor: L
#本文件为神经网络模型应用文件
#读取本目录下保存为GRU.pkl的模型文件，用于标记父目录中dataset目录下data文件夹中的数据文件
#标记后的文件保存在dataset目录中GRU_data文件夹中
'''
import os
import glob
import pandas as pd
import torch 
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
def mkdr(path):
    isExisted = os.path.exists(path)
    if not isExisted:
        os.makedirs(path)
    return path
#加载模型
model = torch.load('GRU.pkl')
criterion = nn.CrossEntropyLoss()
#读取所有数据文件并在原始文件中标记预测结果，保存到dataset/GRU_data/文件夹中
FILE_LST = glob.glob('../dataset/final_data/*.csv')
for file_path in FILE_LST:
    file_name = os.path.basename(file_path)
    df = pd.read_csv(file_path)
    df = df[:-48]
    df = df.set_index(['date', 'seqNo'])
    arr = np.array(df)
    data = arr[:,:-1]
    label = arr[:,-1]
    data = Variable(torch.from_numpy(data).float().view(-1,len(df),in_dim)).cuda()
    label = Variable(torch.from_numpy(label).long()).cuda()
    out = model(data)
    loss = criterion(out[0], label)
    print(loss)
    prediction = torch.max(F.softmax(out[0],dim=1), 1)[1].cpu()
    prediction = prediction.numpy()
    prediction = pd.Series(prediction)
    accr = df['labels'] - prediction
    accr =  accr.map(lambda x: 1 if x == 0 else 0)
    accr_rate = accr.sum()/len(accr)
    print(accr_rate)
    NANS = pd.Series(np.zeros(48))
    prediction = prediction.append(NANS)
    df = pd.read_csv('../dataset/data'+file_name)
    df['GRUlabels'] = prediction
    file_dir = mkdr("../dataset/GRU_data/")  
    df.to_csv(file_dir + file_name) 
    print(file_name + "文件处理完成")
