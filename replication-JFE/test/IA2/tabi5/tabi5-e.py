import os
import pandas as pd
import datetime as dt
from scipy.stats import f
from scipy import linalg
from numpy.linalg import pinv
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd 
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.api import OLS

case = "Gyr_1981_2000__2001_2020_num_iter_9_boost_20"
folder = "P-Tree-e"

if not os.path.exists("./tmp/"+folder+"/"):
    print("make fir. \n ")
    os.mkdir("./tmp/"+folder+"/")

start = '1981-01-01' 
split = '2000-12-31' 
end = '2020-12-31' 

start = pd.to_datetime(start)
split = pd.to_datetime(split)
end = pd.to_datetime(end)

FF = pd.read_csv("../../../data/FactorsMonthly_202312.csv", index_col=0) /100
FF.index = pd.date_range(start="19630731",end="20220531",freq='M')
FF['Const'] = 1
FF = FF.loc[start:split]
RF = FF['RF']
FF[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].to_csv("./tmp/"+folder+"/FF.csv")

Q5 = pd.read_csv("../../../data/Q5/q5_factors_monthly_2021.csv")
Q5.index = pd.date_range('1967-01-31', '2021-12-31', freq='M')
Q5.columns = ['year','month','rf','Q4MKT','Q4ME','Q4IA','Q4ROE','Q4EG']
Q5 = Q5[['Q4MKT','Q4ME','Q4IA','Q4ROE','Q4EG']]/100
Q5 = Q5.loc[start:end]
Q5_ins = Q5.loc[start:split]

def get_FF_factor(factor, FF):
    if factor == 'CAPM':
        F = FF[['Const','Mkt-RF']]
    elif factor == 'FF3':
        F = FF[['Const','Mkt-RF','SMB', 'HML']]
    elif factor == 'FF5':
        F = FF[['Const','Mkt-RF','SMB', 'HML', 'RMW', 'CMA']]
    elif factor == 'Q5':
        F = Q5
    return F

## get ptree portfolios

def get_PTREE_asset(i):
    R1 = pd.read_csv("../"+folder+"/"+case+"_portfolio_fit%s.csv"%i,index_col=0)
    R1.index = pd.date_range(start,split,freq='M')

    return R1

port_tree1 = get_PTREE_asset(1)
port_tree1.to_csv("./tmp/"+folder+"/PT1.csv")

port_tree2 = get_PTREE_asset(2)
port_tree3 = get_PTREE_asset(3)
port_tree4 = get_PTREE_asset(4)
port_tree5 = get_PTREE_asset(5)

port_tree6 = get_PTREE_asset(6)
port_tree7 = get_PTREE_asset(7)
port_tree8 = get_PTREE_asset(8)
port_tree9 = get_PTREE_asset(9)
port_tree10 = get_PTREE_asset(10)

port_tree11 = get_PTREE_asset(11)
port_tree12 = get_PTREE_asset(12)
port_tree13 = get_PTREE_asset(13)
port_tree14 = get_PTREE_asset(14)
port_tree15 = get_PTREE_asset(15)

port_tree16 = get_PTREE_asset(16)
port_tree17 = get_PTREE_asset(17)
port_tree18 = get_PTREE_asset(18)
port_tree19 = get_PTREE_asset(19)
port_tree20 = get_PTREE_asset(20)


port_tree_top5 =         port_tree1.T.append(port_tree2.T).append(port_tree3.T).append(port_tree4.T).append(port_tree5.T)

port_tree_top5 = port_tree_top5.T
port_tree_top5.to_csv("./tmp/"+folder+"/PT_top5.csv")


port_tree_top10 =         port_tree1.T.append(port_tree2.T).append(port_tree3.T).append(port_tree4.T).append(port_tree5.T).append(port_tree6.T).append(port_tree7.T).append(port_tree8.T).append(port_tree9.T).append(port_tree10.T)

port_tree_top10 = port_tree_top10.T
port_tree_top10.to_csv("./tmp/"+folder+"/PT_top10.csv")

port_tree_top15 =         port_tree1.T.append(port_tree2.T).append(port_tree3.T).append(port_tree4.T).append(port_tree5.T).append(port_tree6.T).append(port_tree7.T).append(port_tree8.T).append(port_tree9.T).append(port_tree10.T).append(port_tree11.T).append(port_tree12.T).append(port_tree13.T).append(port_tree14.T).append(port_tree15.T)

port_tree_top15 = port_tree_top15.T
port_tree_top15.to_csv("./tmp/"+folder+"/PT_top15.csv")

port_tree_top20 =         port_tree1.T.append(port_tree2.T).append(port_tree3.T).append(port_tree4.T).append(port_tree5.T).append(port_tree6.T).append(port_tree7.T).append(port_tree8.T).append(port_tree9.T).append(port_tree10.T).append(port_tree11.T).append(port_tree12.T).append(port_tree13.T).append(port_tree14.T).append(port_tree15.T).append(port_tree16.T).append(port_tree17.T).append(port_tree18.T).append(port_tree19.T).append(port_tree20.T)

port_tree_top20 = port_tree_top20.T
port_tree_top20.to_csv("./tmp/"+folder+"/PT_top20.csv")


port_tree_6_10 =        port_tree6.T                .append(port_tree7.T).append(port_tree8.T).append(port_tree9.T).append(port_tree10.T)

port_tree_6_10 = port_tree_6_10.T
port_tree_6_10.to_csv("./tmp/"+folder+"/PT_top6_10.csv")


port_tree_11_15 =      (port_tree11.T).append(port_tree12.T).append(port_tree13.T).append(port_tree14.T).append(port_tree15.T)

port_tree_11_15 = port_tree_11_15.T
port_tree_11_15.to_csv("./tmp/"+folder+"/PT_top11_15.csv")


port_tree_16_20 =      (port_tree16.T).append(port_tree17.T).append(port_tree18.T).append(port_tree19.T).append(port_tree20.T)

port_tree_16_20 = port_tree_16_20.T
port_tree_16_20.to_csv("./tmp/"+folder+"/PT_top16_20.csv")


## get bi-sort


list_bi_sort = ['AC','BETA','BM','INV','LTR','MOM','NI','OP','RESVAR','STR','VAR']
print(len(list_bi_sort))


def get_bi_sort(list_bi_sort):
    df_R = pd.DataFrame()
    for i in list_bi_sort:
        print(i)

        R = pd.read_csv("../../../data/download_portfolios/sort25/25_ME_%s.csv"%i,index_col=0) / 100
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
port_bi.to_csv("./tmp/"+folder+"/Bi.csv")


## get ME/BM25

port_mebm = pd.read_csv("../../../data/download_portfolios/sort25/25_ME_BM.csv",index_col=0) / 100
port_mebm.index = [str(i)+"01" for i in port_mebm.index]
port_mebm.index = pd.to_datetime(port_mebm.index)
port_mebm.index = port_mebm.index+MonthEnd(0)
port_mebm = port_mebm.loc[start:split]
for i in port_mebm.columns:
    port_mebm[i] = port_mebm[i]-RF
port_mebm.head()
port_mebm.to_csv("./tmp/"+folder+"/mebm.csv")


## get uni-sort

list_uni_sort = ['AC','BETA','BM','CFP','DP','EP','INV','LTR','ME','MOM','NI','OP','RESVAR','STR','VAR']
print(len(list_uni_sort))


def get_uni_sort(list_uni_sort):
    df_R = pd.DataFrame()
    for i in list_uni_sort:
        print(i)

        R = pd.read_csv("../../../data/download_portfolios/sort10/10_%s.CSV"%i,index_col=0) / 100
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
port_uni.to_csv("./tmp/"+folder+"/Uni.csv")


## industry

R = pd.read_csv("../../../data/download_portfolios/49_IND.CSV",index_col=0) / 100

R.index = [str(i)+"01" for i in R.index]
R.index = pd.to_datetime(R.index)
R.index = R.index+MonthEnd(0)
R = R.loc[start:end]

for i in R.columns:
    R[i] = R[i]-RF
    
port_ind49 = R.loc[start:split]
port_ind49.to_csv("./tmp/"+folder+"/ind49.csv")


## GRS

def myGRS_plus_rmse(F,R):

    F1 = F.drop('Const',axis=1)
    
    T = F1.shape[0]  # number of time-series observations
    N = R.shape[1]   # number of portfolios
    K = F1.shape[1]  # number of factors

    # OLS
    B = pinv(F.T.dot(F)).dot(F.T.dot(R))
    B = pd.DataFrame(B)
    B.index = F.columns
    B.columns = R.columns

    hatR = F.dot(B)
    Res = hatR - R
    rmse = ((Res**2).mean()**0.5).mean()

    Alpha = B.loc['Const']
    # Alpha

    R2 = 100 * (1 - Res.var()/R.var())
    # print(R2)

    # dividing the GRS equation into 3 sections a, b and c to simplyfy

    # Part a
    a = (T/N) * ((T-N-K)/(T-K-1))
    
    # Part b
    # omega hat should be a K x K matrix (verified and True)
    E_f = F.mean()
    omega_hat = 1/(T-1)*((F-E_f).T).dot(F-E_f)
    # b should be a scalar (verified and True)
    omega_hat_inv = pinv(omega_hat)  # pseudo-inverse
    b = 1 + E_f.T.dot(omega_hat_inv).dot(E_f)
    b_inv = 1/b

    # Part c
    # sigma hat should be a N x N matrix (verified and True)
    sigma_hat = 1/(T-K-1)*Res.T.dot(Res)
    sigma_hat_inv = pinv(sigma_hat)  # pseudo-inverse
    alpha_hat = Alpha
    c = alpha_hat.T.dot(sigma_hat_inv).dot(alpha_hat)
    
    
    # Putting the 3 GRS parts together
    GRS = a*b_inv*c
    pvalue = f.sf(GRS, N , T - N - K)
    
    return (GRS, pvalue, 100*Alpha.abs().mean(), (10000*Alpha**2).mean(), rmse*100, R2.mean())

myGRS_plus_rmse(FF[['Const','Mkt-RF','SMB', 'HML', 'RMW', 'CMA']],port_ind49)




# Regression
def myGRS_plus_rmse_alpha_proportion(F,R):
    
    R.columns = range(len(R.columns))

    F1 = F.drop('Const',axis=1)
    
    T = F1.shape[0]  # number of time-series observations
    N = R.shape[1]   # number of portfolios
    K = F1.shape[1]  # number of factors

    # OLS
    B = pinv(F.T.dot(F)).dot(F.T.dot(R))
    B = pd.DataFrame(B)
    B.index = F.columns
    B.columns = R.columns

    hatR = F.dot(B)
    Res = hatR - R
    rmse = ((Res**2).mean()**0.5).mean()

    Alpha = B.loc['Const']
    # Alpha

    R2 = 100 * (1 - Res.var()/R.var())
    # print(R2)

    # dividing the GRS equation into 3 sections a, b and c to simplyfy

    # Part a
    a = (T/N) * ((T-N-K)/(T-K-1))
    
    # Part b
    # omega hat should be a K x K matrix (verified and True)
    E_f = F.mean()
    omega_hat = 1/(T-1)*((F-E_f).T).dot(F-E_f)
    # b should be a scalar (verified and True)
    omega_hat_inv = pinv(omega_hat)  # pseudo-inverse
    b = 1 + E_f.T.dot(omega_hat_inv).dot(E_f)
    b_inv = 1/b

    # Part c
    # sigma hat should be a N x N matrix (verified and True)
    sigma_hat = 1/(T-K-1)*Res.T.dot(Res)
    sigma_hat_inv = pinv(sigma_hat)  # pseudo-inverse
    alpha_hat = Alpha
    c = alpha_hat.T.dot(sigma_hat_inv).dot(alpha_hat)
    
    
    
    # Putting the 3 GRS parts together
    GRS = a*b_inv*c
    pvalue = f.sf(GRS, N , T - N - K)
    
    # t-stat of alpha
    list_alpha=[]
    list_alphat=[]
    for i in tqdm(R.columns):
        Y = R[i].values
        X = sm.add_constant(F1)
        model = OLS(Y,X)
        results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
        list_alpha.append(results.params['const']*100)
        list_alphat.append(results.tvalues['const'])

    print(list_alpha)
    print(list_alphat)
    
    return (GRS, pvalue,             100*Alpha.abs().mean(), 
            
            (10000*Alpha**2).mean(), \
            
            100*(Alpha**2).mean()**0.5, \
            
            100*rmse, \
            R2.mean(), \
            np.mean(np.abs(list_alphat)>1.64)*100, \
            np.mean(np.abs(list_alphat)>1.96)*100, \
            np.mean(np.abs(list_alphat)>2.58)*100
           )

myGRS_plus_rmse_alpha_proportion(FF[['Const','Mkt-RF','SMB', 'HML', 'RMW', 'CMA']],port_tree1)



for factor in ['FF5']:
    
    df = pd.DataFrame([])
    i=0
        
    for port in [
                port_tree1,
                port_tree_top5,
                port_tree_6_10,
                port_tree_11_15,
                port_tree_16_20,
                port_tree_top20
                ]:

        i=i+1
        R = port
        N = port.shape[1]
    
        # print(factor)
        F = get_FF_factor(factor, FF)
        (grs,p,a,a2,ra2,rmse,r2, pct_164, pct_196, pct_258) = myGRS_plus_rmse_alpha_proportion(F,R)
        df[str(i)+"-"+factor] = [i,factor,N,grs,p,a, ra2, r2, pct_164, pct_196, pct_258]
        
    df = df.T
    df.columns = ['Asset','Factor','N','GRS','p','100xAbaAlp','100xRoot_SqdAlp','AvgR2', "Pct_164", "Pct_196", "Pct_258"]
    
    df.index = ["PTree1",
                "PTree1-5",
                "PTree6-10",
                "PTree11-15",
                "PTree16-20",
                "PTree1-20"
                ]
    df.to_csv("tabi5-e.csv")
