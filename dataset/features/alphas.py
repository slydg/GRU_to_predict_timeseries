import numpy as np
import pandas as pd 

#设定基础函数
def get_delta(x,N):
    lst=[]
    for i in range(1,N+2):
        lst.append(x)
        x=x.diff()
    return lst[-1]

def rank(array):
    s = pd.Series(array)
    return s.rank(ascending=False)[len(s)-1]

#单只股票的time series分析指标
def get_alpha1(df):
    df['a1']=(df['high']-df['low'])/abs(df['close']-df['open'])
    return df['a1']

def get_alpha2(df):
    df['a2']=(df['high']-df['close'])/(df['close']-df['low'])
    return df['a2']

def get_alpha3(df):
    df['a3']=(df['high']-df['open'])/(df['open']-df['close'])
    return df['a3']

# Alpha#1: (SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.)
def get_alpha4(df):
    filter4_1=df['returns'].rolling(window=20).std().where(df['returns']<0)
    df['ralpha4']=filter4_1.fillna(df['close'])
    return df['ralpha4']**2

#Alpha#3:correlation(high,volume,5)
def get_alpha5(df):
    return df['high'].rolling(window=5).corr(df['volume'])

#Alpha#4:correlation(open,volume,10)
def get_alpha6(df):
    return df['open'].rolling(window=10).corr(df['volume'])
    
#Alpha#5:delta(sum(open,5)*sum(return,5),10)
def get_alpha7(df):
    return get_delta(df['open'].rolling(window=5).sum()*df['returns'].rolling(window=5).sum(),10)

#Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))) 
def get_alpha8(df):
    filter9_1=df['close'].diff().where((df['close'].diff().rolling(window=5).min())<0)
    filter9_2=filter9_1.fillna(df['close'].diff().where((df['close'].diff().rolling(window=5).max())>0))
    return filter9_2.fillna(-df['close'].diff())

#Alpha#12: (delta(volume, 1)) * (-1 * delta(close, 1))
def get_alpha9(df):
    return -1*df['volume'].diff()*df['close'].diff()

#Alpha#18: stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)
def get_alpha10(df):
    return abs(df['close']-df['open']).rolling(window=5).std()+(df['close']-df['open'])+get_alpha7(df)

#Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0) 
def get_alpha11(df):
    df['alpha12']=-1*get_delta(df['high'],2).where(df['high'].rolling(window=20).mean()<df['high'])
    return df['alpha12'].fillna(0)

#Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3))) 
def get_alpha12(df):
    nofilter13=df['close'].rolling(window=100).min()-df['close']
    filter13_1=nofilter13.where(get_delta(df['close'].rolling(window=100).mean(),100)/(df['close']-get_delta(df['close'],100))<0.05)
    filter13_2=filter13_1.fillna(nofilter13.where(get_delta(df['close'].rolling(window=100).mean(),100)/(df['close']-get_delta(df['close'],100))==0.05))
    return filter13_2.fillna(-get_delta(df['close'],3))

#Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)) 
def get_alpha13(df):
    return df['volume'].rolling(window=5).apply(func=rank).rolling(window=5).corr(df['high'].rolling(window=5).apply(func=rank)).rolling(window=3).max()

#Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32))) 
def get_alpha14(df):
    return df['volume'].rolling(window=32).apply(func=rank)*(1-(df['close']+df['high']-df['low']).rolling(window=32).apply(func=rank))*(1-df['returns'].rolling(window=32).apply(func=rank))

#Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))  
def get_alpha15(df):
    return df['volume']/df['close']*df['volume'].rolling(window=20).mean().rolling(window=20).apply(func=rank)*-get_delta(df['close'],7).rolling(window=8).apply(func=rank)

#Alpha#55: correlation((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), volume, 6)
def get_alpha16(df):
    return df['volume'].rolling(window=6).corr((df['close']-df['low'].rolling(window=12).min())/(df['high'].rolling(window=12).max()-df['low'].rolling(window=12).min()))

#Alpha#101: ((close - open) / ((high - low) + .001)) 
def get_alpha17(df):
    return (df['close']-df['open'])/(df['high']-df['low']+0.01)

#Alpha#28: (correlation(adv20, low, 5) + ((high + low) / 2)) - close
def get_alpha18(df):
    return df['low'].rolling(window=5).corr(df['close']*df['volume'].rolling(window=20).mean())+(df['high']+df['low'])/2-df['close']

#EMV:A=（今日最高+今日最低）/2;B=（前日最高+前日最低）/2;C=今日最高-今日最低;2.EM=（A-B）*C/今日成交额;3.EMV=N日内EM的累和;4.MAEMV=EMV的M日简单移动平均.参数N为14，参数M为9
def get_alpha19(df):
    return get_delta((df['high']+df['low'])/2,1).rolling(window=14).sum().rolling(window=9).mean()

#MTMMA动量指标
def get_alpha20(df):
    return get_delta(df['close'],12).rolling(window=6).mean()

#横截面分析指标
#Alpha#41: (((high * low)^0.5) - vwap) 

#Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))  

#Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796)) 

#Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))  
