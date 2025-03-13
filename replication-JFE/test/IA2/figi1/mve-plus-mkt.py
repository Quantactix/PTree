import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.linalg import inv, pinv
from tqdm import tqdm
from pandas.tseries.offsets import MonthEnd 
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

start = '1981-01-31'
split = '2020-12-31'
end   = '2020-12-31'

start = pd.to_datetime(start)
split = pd.to_datetime(split)
end = pd.to_datetime(end)

folder = "../P-Tree-d/"
case='Gyr_1981_2020_num_iter_9_boost_20' 

f_ins = pd.read_csv(folder+case+"_bf_train.csv", index_col=0)

def draw_with_MKT(f_ins, idx, N, scale):
    
    port_ins = pd.DataFrame()
    W = pd.DataFrame()
    
    p = f_ins[f_ins.columns[:idx]]
    n = p.shape[1]

    # maxSR
    w = pinv(p.cov()).dot(p.mean())
    w=w/w.sum()
    port_ins['maxSR'] = p.dot(w)
    W['maxSR'] = w

    # minVar
    w = pinv(p.cov()).dot(np.ones(n))
    w=w/w.sum() * scale
    port_ins['minVar'] = p.dot(w)
    W['minVar'] = w

    N = 100
    scale = np.linspace(0,20,N)
    for i in range(N):
        W[i] = scale[i] * W['maxSR'] + (1-scale[i]) * W['minVar']

    for i in range(N):
        port_ins[i] = p.dot(W[i].values)

    df = pd.DataFrame()
    df['mean'] = port_ins.mean()
    df['std'] = port_ins.std()
    df

    df_two_fund = df.iloc[:2,]*100
    df_line = df.iloc[2:,]*100
    
    r=idx / N_F
    g=1-idx / N_F
    b=1-idx / N_F

    plt.plot(df_line['std'], df_line['mean'], color=[r,g,b], label='MKT+%s'%(idx-1), alpha=1)#*idx/N_F + 0.0)
    plt.scatter(df_two_fund['std']['maxSR'], df_two_fund['mean']['maxSR'], color=[r,g,b], marker='.', s=100, zorder=100)
    
def draw_bench(p, N, lab, c, style):
    
    port_ins = pd.DataFrame()
    W = pd.DataFrame()
    
    n = p.shape[1]

    # maxSR
    w = pinv(p.cov()).dot(p.mean())
    w=w/w.sum()
    port_ins['maxSR'] = p.dot(w)
    W['maxSR'] = w

    # minVar
    w = pinv(p.cov()).dot(np.ones(n))
    w=w/w.sum()
    port_ins['minVar'] = p.dot(w)
    W['minVar'] = w

    scale1 = np.linspace(0,20,N)
    for i in range(N):
        W[i] = scale1[i] * W['maxSR'] + (1-scale1[i]) * W['minVar']

    for i in range(N):
        port_ins[i] = p.dot(W[i].values)

    df = pd.DataFrame()
    df['mean'] = port_ins.mean()
    df['std'] = port_ins.std()
    df

    df_two_fund = df.iloc[:2,]*100
    df_line = df.iloc[2:,]*100
    
    plt.plot(df_line['std'], df_line['mean'], color=c, label=lab, alpha=1, linestyle=style)#*idx/N_F + 0.0)
    plt.scatter(df_two_fund['std']['maxSR'], df_two_fund['mean']['maxSR'], color=c, marker=".", s=100, zorder=100)

def draw_mve_asset(p, N, lab, c, style, scale):
    
    port_ins = pd.DataFrame()
    W = pd.DataFrame()
    
    n = p.shape[1]

    # maxSR
    w = pinv(p.cov()).dot(p.mean())
    w=w/np.abs(w).sum() * scale

    port_ins['maxSR'] = p.dot(w)
    W['maxSR'] = w

    # minVar
    w = pinv(p.cov()).dot(np.ones(n))
    w=w/np.abs(w).sum() * scale
    port_ins['minVar'] = p.dot(w)
    W['minVar'] = w

    scale = np.linspace(0,1,N)
    for i in range(N):
        W[i] = scale[i] * W['maxSR'] + (1-scale[i]) * W['minVar']

    for i in range(N):
        port_ins[i] = p.dot(W[i].values)

    df = pd.DataFrame()
    df['mean'] = port_ins.mean()
    df['std'] = port_ins.std()
    
    #### find the empirical MinVar
    new_port_ins = pd.DataFrame()
    new_W = pd.DataFrame()

    idx_MinVar = df[df['std']==(df.min()['std'])].index[0]
    new_W['minVar'] = W[idx_MinVar]
    new_W['maxSR'] = W['maxSR']
    new_port_ins['minVar'] = port_ins[idx_MinVar]
    new_port_ins['maxSR'] = port_ins['maxSR']
    
    scale = np.linspace(0,10,N)
    for i in range(N):
        new_W[i] = scale[i] * new_W['maxSR'] + (1-scale[i]) * new_W['minVar']

    for i in range(N):
        new_port_ins[i] = p.dot(new_W[i].values)

    df = pd.DataFrame()
    df['mean'] = new_port_ins.mean()
    df['std'] = new_port_ins.std()

    df_two_fund = df.iloc[:2,]*100
    df_line = df.iloc[2:,]*100
    
    plt.plot(df_line['std'], df_line['mean'], color=c, label=lab, alpha=1, linestyle=style)#*idx/N_F + 0.0)
    plt.scatter(df_two_fund['std']['maxSR'], df_two_fund['mean']['maxSR'], color=c, marker=".", s=100, zorder=100)

## read factors

FF = pd.read_csv("../../data/FactorsMonthly_202205.csv", index_col=0) /100
FF.index = pd.date_range(start="19630731",end="20220531",freq='M')
RF = FF['RF']

FF = FF.loc[start:end]
FF_ins = FF.loc[start:split]
FF5 = FF_ins[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]

Q5 = pd.read_csv("../../data/Q5/q5_factors_monthly_2021.csv") #/100
Q5.index = pd.date_range('1967-01-31', '2021-12-31', freq='M')
Q5.columns = ['year','month','rf','Q4MKT','Q4ME','Q4IA','Q4ROE','Q4EG']
Q5 = Q5[['Q4MKT','Q4ME','Q4IA','Q4ROE','Q4EG']]/100
Q5 = Q5.loc[start:end]
Q5_ins = Q5.loc[start:split]

RP = pd.read_csv("../../data/RP-PCA/rppca_1981_2020.csv", header=None)
RP.index = pd.date_range(start.strftime("%Y-%m-%d"), split.strftime("%Y-%m-%d"), freq='M')
RP_ins = RP.loc[start.strftime("%Y"):split.strftime("%Y")]
RP_ins = RP_ins[RP_ins.columns[:5]]

IP_ins = pd.read_csv("../../data/IPCA/ipca_1981_2020.csv", index_col=0)
IP_ins = IP_ins[IP_ins.columns[:5]]

## read test assets

list_uni_sort = ['AC','BETA','BM','CFP','DP','EP','INV','LTR','ME','MOM','NI','OP','RESVAR','STR','VAR']
def get_uni_sort(list_uni_sort):
    df_R = pd.DataFrame()
    for i in list_uni_sort:
        R = pd.read_csv("../../data/download_portfolios/sort10/10_%s.CSV"%i,index_col=0) / 100
        R.index = [str(i)+"01" for i in R.index]
        R.index = pd.to_datetime(R.index)
        R.index = R.index+MonthEnd(0)
        R = R.loc[start:end]
        R.columns = [i+"-"+j for j in R.columns]
        
        df_R = df_R.T.append(R.T).T
        
    for i in df_R.columns:
        df_R[i] = df_R[i]-RF
    
    return df_R

port_uni = get_uni_sort(list_uni_sort).loc[start:split]

port_mebm = pd.read_csv("../../data/download_portfolios/sort25/25_ME_BM.csv",index_col=0) / 100
port_mebm.index = [str(i)+"01" for i in port_mebm.index]
port_mebm.index = pd.to_datetime(port_mebm.index)
port_mebm.index = port_mebm.index+MonthEnd(0)
port_mebm = port_mebm.loc[start:split]
for i in port_mebm.columns:
    port_mebm[i] = port_mebm[i]-RF

list_bi_sort = ['AC','BETA','BM','INV','LTR','MOM','OP','RESVAR','STR','VAR','NI']
def get_bi_sort(list_bi_sort):
    df_R = pd.DataFrame()
    for i in list_bi_sort:
        R = pd.read_csv("../../data/download_portfolios/sort25/25_ME_%s.csv"%i,index_col=0) / 100
        R.index = [str(i)+"01" for i in R.index]
        R.index = pd.to_datetime(R.index)
        R.index = R.index+MonthEnd(0)
        R = R.loc[start:end]
        R.columns = [i+"-"+j for j in R.columns]
        
        df_R = df_R.T.append(R.T).T
        
    print(df_R.head())

    for i in df_R.columns:
        df_R[i] = df_R[i]-RF
    
    return df_R

port_bi = get_bi_sort(list_bi_sort).loc[start:split]


R = pd.read_csv("../../data/download_portfolios/49_IND.CSV",index_col=0) / 100

R.index = [str(i)+"01" for i in R.index]
R.index = pd.to_datetime(R.index)
R.index = R.index+MonthEnd(0)
R = R.loc[start:end]

for i in R.columns:
    R[i] = R[i]-RF
    
port_ind49 = R.loc[start:split]

## plot it

plt.figure(figsize=(10,10))
plt.xlim((0,1.2))
plt.ylim((0,2))

N_F = f_ins.shape[1]
N = 1000

for idx in tqdm(range(2,N_F+1)):
    print(idx)
    draw_with_MKT(f_ins,idx,N,1)

draw_bench(FF5,N,"FF5",'black','solid')
draw_bench(Q5,N,"Q5",'black','dashed')
draw_bench(RP_ins,N,"RP5",'black','dashdot')
draw_bench(IP_ins,N,"IP5",'black','dotted')

draw_mve_asset(port_uni,N,"Uni-Sort",'purple','solid', 10)
draw_mve_asset(port_bi,N,"Bi-Sort",'purple','dashed', 15)
draw_mve_asset(port_mebm,N,"ME/BM",'purple','dashdot', 1.5)
draw_mve_asset(port_ind49,N,"Ind49",'purple','dotted', 1)

plt.xlabel('Std (%)')
plt.ylabel('Mean (%)')
plt.legend(loc='upper left', fontsize='small')
plt.savefig("MVE-plus-mkt.pdf",bbox_inches = 'tight')

# plt.show()
plt.close()
