'''
#edit_time: 2018-2-11
#editor: L
#本文件用于为数据降维以及打标签
#数据源文件保存在本文件同层级的adj_data文件夹中，处理后的数据文件保存在本文件同层级的final_data文件夹中
'''
import os
import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
#以下部分定义本文件中用到的函数
#创建写入文件夹文件
def mkdr(path):
    isExisted = os.path.exists(path)#判断路径是否存在
    if not isExisted:#若不存在，创建路径
        os.makedirs(path)
    return path
#打上标签，弃用标签方法，每一周期标记下一天涨跌
# def get_labels(df):
#     index = df.index.get_level_values(0)
#     index = index.tolist()
#     index = sorted(list(set(index)))
#     close = df['1']
#     for i in range(len(index)-1):
#         item1 = index[i] 
#         item2 = index[i + 1]
#         data1 = close[int(item1)]
#         data1 = data1.tolist()[-1]
#         data2 = close[int(item2)]
#         data2 = data2.tolist()[-1]
#         if (data2 - data1) / data1 < -0.0005:
#             df.set_value(item1,'labels',0)
#         elif (data2 - data1) / data1 > 0.0005:
#             df.set_value(item1, 'labels', 2)
#         else:
#             df.set_value(item1, 'labels', 1)
#     return df['labels']
#打上标签，使用中标签方法，每一周期标记下48个周期涨跌
def get_labels(df):
    close = df['close']
    clprice1 = close[:-48].reset_index(drop=True)
    clprice2 = close[48:].reset_index(drop=True)
    label = (clprice2 - clprice1)/(clprice1 + 0.0001)
    print(label)
    label = label.map(lambda x: 0 if x <= -0.05 else 1 if -0.05 < x <= -0.005 else 2 if -0.005 < x <= 0.005 else 3 if 0.005 < x <= 0.05 else 4)
    NANS = pd.Series(np.zeros(48))
    label = label.append(NANS)
    print(label)
    return label

#获取PCA训练集
def get_randlines():
    trainlst = []
    labellst = []
    linelst1 = []
    for file_path in FILE_LST:
        print(file_path)
        with open(file_path) as f:
            label = f.readline()
            labellst = label.split(',')
            for line in f:
                if np.random.randn() > 3:
                    linelst1 = line.split(',')
                    trainlst.append(float(i) for i in linelst1)
                else:
                    continue
    for trainunit in trainlst:
        if trainunit == labellst:
            trainlst.remove(trainunit)
    train = pd.DataFrame(data=trainlst, columns=labellst)
    return train

#main function
if __name__ == "__main__":
    FILE_LST = glob.glob('./adj_data/*.csv')
    train = get_randlines().drop(['date', 'seqNo'], axis=1)
    print(train)
    #用训练集训练
    pca = PCA(n_components=20)
    train_pca = pca.fit(train)
    print(train_pca.explained_variance_ratio_)
    print(sum(train_pca.explained_variance_ratio_))
    train = []
    for file_path in FILE_LST:
        file_name = os.path.basename(file_path)
        origin_file = './data/' + file_name
        df = pd.read_csv(file_path)
        origin_df = pd.read_csv(origin_file)
        dt = df['date']
        ds = df['seqNo']
        df = df.set_index(['date', 'seqNo'])
        np_pca = train_pca.transform(df)#对数据进行降维处理
        df_pca = pd.DataFrame(np_pca)
        dl = get_labels(origin_df)
        df_pca['labels'] = dl.values#在降维后的数据上标上标签
        df_pca['date'] = dt
        df_pca['seqNo'] = ds
        df_pca = df_pca.set_index(['date', 'seqNo'])#设置索引
        print(file_name + "文件处理完成")
        file_dir = mkdr("./final_data/")  # 创建目标路径
        df_pca.to_csv(file_dir + file_name)  # 把处理过的文件存入pca_data文件夹
