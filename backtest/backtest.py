'''
#edit_time: 2018-2-23
#editor: C
#本文件用于回测模型输出，回测结果可用于衡量模型是否有效
'''
import pandas as pd
import numpy as np
import glob
import os
import random

rvntotlst=[]
ydtotlst=[]
winctot=0
losctot=0
tradectot=0
cnt=0
def price_adj():
    praj = pd.read_csv('../dataset/fq.csv')
    praj = praj.set_index(['index']).T
    return praj
def adj(df, praj):
    date = praj.index.values.tolist()
    for item in date:
        try:
            df[int(item)] = df[int(item)] * praj[item]
        except:
            continue
    return df
def get_features(df, praj):
	df['close'] = adj(df['close'], praj)/10000
	df['open'] = adj(df['open'], praj)/10000
	df['high'] = adj(df['high'], praj)/10000
	df['low'] = adj(df['low'], praj)/10000
	return df
praj= price_adj()
FILE_LST = glob.glob('./GRU_data/*.csv')
fnb = int(0.1 * len(FILE_LST))
SL_LST = random.sample(FILE_LST, fnb)
random.shuffle(SL_LST)
for file_path in SL_LST:
	state=0	#每只股票初始化
	rvnlst=[]
	ydlst=[]
	winc=0
	losc=0
	tradec=0
	file_name = os.path.basename(file_path)
	print("*****正在处理"+file_name+"*****")
	cnt+=1
	data = pd.read_csv(file_path)
	#处理复权因子
	if int(os.path.splitext(file_name)[0]) >= 600000:
		cl_name = os.path.splitext(file_name)[0] + '.SH'
	else:
		cl_name = os.path.splitext(file_name)[0] + '.SZ'
	try:
		data = get_features(data, praj[cl_name])#获取所有特征
	except:
		continue
	close=data['close'] 
	clprice1 = close[:-48].reset_index(drop=True)
	clprice2 = close[48:].reset_index(drop=True)
	data['change'] = (clprice2 - clprice1)/(clprice1 + 0.0001)
	df=data[:-48]		#最后48个change无效
	for date in df['date'].unique():
		zt=df['change'].where(df['date']==date)>0.1
		dt=df['change'].where(df['date']==date)<-0.1
		if True in list(zt) or True in list(dt):
			continue	#如果出现涨跌停，直接进入下一天
		else:
			for i in range(len(list(df['GRUlabels'].where(df['date']==date).dropna()))):
				if state==0:
					if list(df['GRUlabels'].where(df['date']==date).dropna())[i]==3.0 or list(df['GRUlabels'].where(df['date']==date).dropna())[i]==4.0:
						if tradec==0:
							pricesell=list(close.where(df['date']==date).dropna())[i]
							pricebuy=list(close.where(df['date']==date).dropna())[i]
							print("第一次买入")
							state=1
							continue
						elif pricesell-list(close.where(df['date']==date).dropna())[i]>-0.05:     #交易风险控制参数设置为－0.05
							pricebuy=list(close.where(df['date']==date).dropna())[i]
							state=1
							print("买入")
							continue
				else:
					if list(df['GRUlabels'].where(df['date']==date).dropna())[i]==0.0 or list(df['GRUlabels'].where(df['date']==date).dropna())[i]==1.0:
						if list(close.where(df['date']==date).dropna())[i]-pricebuy>0.2:      #期望的收益参数设置为0.2
							pricesell=list(close.where(df['date']==date).dropna())[i]
							state=0
							winc+=1
							tradec+=1
							rvn=pricesell-pricebuy
							yd=(pricesell-pricebuy)/pricebuy
							print("卖出，收益为"+ str(rvn)+",收益率为"+str(round(yd*100,2))+"%")
							rvnlst.append(rvn)
							rvntotlst.append(rvn)
							ydlst.append(yd)
							ydtotlst.append(yd)
							continue
						if (pricebuy-list(close.where(df['date']==date).dropna())[i])/pricebuy>0.1:       #止损参数设置为0.1
							pricesell=list(close.where(df['date']==date).dropna())[i]
							state=0
							losc+=1
							tradec+=1
							rvn=pricesell-pricebuy
							yd=(pricesell-pricebuy)/pricebuy
							print("卖出，收益为"+ str(rvn)+",收益率为"+str(round(yd*100,2))+"%")
							rvnlst.append(rvn)
							rvntotlst.append(rvn)
							ydlst.append(yd)
							ydtotlst.append(yd)
							continue
	tradectot+=tradec
	winctot+=winc
	losctot+=losc
	print("**本股获利次数"+str(winc)+"\n"+"**本股亏损次数"+str(losc))
	print("**本股获利次数比"+str(round(winc/tradec,2))+"\n"+"**本股亏损次数比"+str(round(losc/tradec,2)))
	print("**本股最大获益"+str(max(rvnlst)))
	print("**本股最大亏损"+str(min(rvnlst)))
	print("**本股平均收益"+str(round(sum(rvnlst)/len(rvnlst),2)))
	print("**本股最大获益率"+str(round(max(ydlst)*100,2))+"%")
	print("**本股最大亏损率"+str(round(min(ydlst)*100,2))+"%")
	print("**本股平均收益率"+str(round(sum(ydlst)*100/len(ydlst),2))+"%")
	print("*****report*****")
	print("共随机测试"+str(cnt)+"股")
	print("**总获利次数"+str(winctot)+"\n"+"**总亏损次数"+str(losctot))
	print("**总获利次数比"+str(round(winctot/tradectot,2))+"\n"+"**总本股亏损次数比"+str(round(losctot/tradectot,2)))
	print("**最大获益"+str(max(rvntotlst)))
	print("**最大亏损"+str(min(rvntotlst)))
	print("**平均收益"+str(round(sum(rvntotlst)/len(rvntotlst),2)))
	print("**最大获益率"+str(round(max(ydtotlst)*100,2))+"%")
	print("**最大亏损率"+str(round(min(ydtotlst)*100,2))+"%")
	print("**平均收益率"+str(round(sum(ydtotlst)*100/len(ydtotlst),2))+"%")