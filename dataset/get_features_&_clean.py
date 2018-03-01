'''
#edit_time: 2018-2-8
#editor: L
#本文件用于读取，处理，储存数据文件
#数据源文件保存在本文件同层级的data文件夹中，处理后的数据文件保存在本文件同层级的adj_data文件夹中
#数据最终结果为包含所有特征并进行标准化和清洗后的数据
'''
import os
import glob
import pandas as pd
import numpy as np
import stockstats
from features import alphas #获取数据特征的包
from sklearn.preprocessing import MinMaxScaler
#以下部分定义本文件中用到的函数
#创建写入文件夹文件
def mkdr(path):
    isExisted = os.path.exists(path)#判断路径是否存在
    if not isExisted:#若不存在，创建路径
        os.makedirs(path)
    return path

def price_adj():
    praj = pd.read_csv('./fq.csv')
    praj = praj.set_index(['index']).T
    return praj

#处理复权因子
def adj(df, praj):
    date = praj.index.values.tolist()
    for item in date:
        try:
            df[int(item)] = df[int(item)] * praj[item]
        except:
            continue
    return df
#获取特征值
def get_features(df, praj):
    df['close'] = adj(df['close'], praj)/10000
    df['open'] = adj(df['open'], praj)/10000
    df['high'] = adj(df['high'], praj)/10000
    df['low'] = adj(df['low'], praj)/10000
    close = df['close']         
    df['returns']=close.diff()
    df['delta_close_2']=df['returns'].diff()
    df['alpha1']=(df['high']-df['low'])/abs(df['close']-df['open']+0.01)
    df['alpha2']=(df['high']-df['close'])/(df['close']-df['low']+0.01)
    df['alpha3']=(df['high']-df['open'])/(df['open']-df['close']+0.01)
    df['alpha4']=alphas.get_alpha4(df)
    df['alpha5']=alphas.get_alpha5(df)
    df['alpha6']=alphas.get_alpha6(df)
    df['alpha7']=alphas.get_alpha7(df)
    df['alpha8']=alphas.get_alpha8(df)
    df['alpha9']=alphas.get_alpha9(df)
    df['alpha10']=alphas.get_alpha10(df)
    df['alpha11']=alphas.get_alpha11(df)
    df['alpha12']=alphas.get_alpha12(df)
    df['alpha13']=alphas.get_alpha13(df)
    df['alpha14']=alphas.get_alpha14(df)
    df['alpha15']=alphas.get_alpha15(df)
    df['alpha16']=alphas.get_alpha16(df)
    df['alpha17']=alphas.get_alpha17(df)
    df['alpha18']=alphas.get_alpha18(df)
    df['alpha19']=alphas.get_alpha19(df)
    df['alpha20']=alphas.get_alpha20(df)
    stock=stockstats.StockDataFrame.retype(df)
# CR indicator, including 5, 10, 20 days moving average
    df['cr']=stock['cr']
    df['cr-mal']=stock['cr-ma1']
    df['cr-mal2']=stock['cr-ma2']
    df['cr-mal3']=stock['cr-ma3']
# KDJ, default to 9 days
    df['kdjk']=stock['kdjk']
    df['kdjd']=stock['kdjd']
    df['kdjj']=stock['kdjj']
# MACD
    df['macd']=stock['macd']
# bolling, including upper band and lower band
    df['boll']=stock['boll']
    df['boll_ub']=stock['boll_ub']
    df['boll-lb']=stock['boll_lb']
# CR MA2 cross up CR MA1 in 20 days count
    df['cr_ma2_cr_ma1']=stock['cr-ma2_xu_cr-ma1_20_c']
# 6 days RSI
    df['rsi_6']=stock['rsi_6']
# 12 days RSI
    df['rsi_12']=stock['rsi_12']
# 10 days WR
    df['wr_10']=stock['wr_10']
# 6 days WR
    df['wr_6']=stock['wr_6']
# CCI, default to 14 days
    df['cci']=stock['cci']
# 20 days CCI
    df['cci_20']=stock['cci_20']
# TR (true range)
    df['tr']=stock['tr']
# ATR (Average True Range)
    df['atr']=stock['atr']
# DMA, difference of 10 and 50 moving average
    df['dma']=stock['dma']
# DMI
# +DI, default to 14 days
    df['pdi']=stock['pdi']
# -DI, default to 14 days
    df['mdi']=stock['mdi']
# DX, default to 14 days of +DI and -DI
    df['dx']=stock['dx']
# ADX, 6 days SMA of DX, same as stock['dx_6_ema']
    df['adx']=stock['adx']
# ADXR, 6 days SMA of ADX, same as stock['adx_6_ema']
    df['adxr']=stock['adxr']
# TRIX, default to 12 days
    df['trix']=stock['trix']
# MATRIX is the simple moving average of TRIX
    df['trix_9_sma']=stock['trix_9_sma']
# VR, default to 26 days
    df['vr']=stock['vr']
# MAVR is the simple moving average of VR
    df['vr_6_sma']=stock['vr_6_sma']
    return df
#清洗数据
def data_cleaning(df):
    #删除stockstats两列值为bool类型
    np.where(df['cr-ma2_xu_cr-ma1_20_c'] is True,1,0)
    np.where(df['cr_ma2_cr_ma1'] is True,1,0)
    #清除剩余的inf和nan
    df=df.replace([np.inf, -np.inf], np.nan)
    df_clean=df.replace(np.nan,0)
    #Minmaxscaler原理：X_std=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))；X_scaled=X_std/(max-min)+min
    scaler=MinMaxScaler(copy=False)
    np_scale=scaler.fit_transform(df_clean)
    df_scale=pd.DataFrame(np_scale)
    return df_scale

#运行主函数
if __name__ == "__main__":
    #数据源文件路径
    FILE_LST = glob.glob('./data/*.csv')
    praj= price_adj()
    #循环读取数据文件
    for file_path in FILE_LST:
        file_name = os.path.basename(file_path)#获取数据源文件文件名
        df = pd.read_csv(file_path)#读取数据源文件保存为名为this_file的dataframe格式
        dt = df['date']
        ds = df['seqNo']
        df = df.set_index(['date', 'seqNo'])
        if len(df.index)<2:
            print(file_name + "内容为空")#排除空文档
            continue
        if int(os.path.splitext(file_name)[0]) >= 600000:
            cl_name = os.path.splitext(file_name)[0] + '.SH'
        else:
            cl_name = os.path.splitext(file_name)[0] + '.SZ'
        try:
            df = get_features(df, praj[cl_name])#获取所有特征
        except:
            continue 
        df = data_cleaning(df)#清洗数据
        df['date'] = dt
        df['seqNo'] = ds
        df = df.set_index(['date', 'seqNo'])
        print(file_name + "文件处理完成")
        file_dir = mkdr("./adj_data/")#创建目标路径
        df.to_csv(file_dir + file_name)#把处理过的文件存入adj_data文件夹
