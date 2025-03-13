
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import statsmodels.api as sm
from numpy.linalg import inv, pinv
import warnings
warnings.filterwarnings("ignore")

case='Gyr_2001_2020__1981_2000_num_iter_9_boost_20' 
folder = "P-Tree-f"

start = '2001-01-01' 
split = '2020-12-31' 
end = '2020-12-31' 

start = pd.to_datetime(start)
split = pd.to_datetime(split)
end = pd.to_datetime(end)

lambda_mean = 0
lambda_cov = 1e-5

folder = "../"+folder+"/"
F = pd.read_csv(folder+case+"_bf_train.csv", index_col=0)

# SR

ls_sr = []
ls_mve_sr = []
ls_mve_mean = []
ls_mve_std = []

for i in range(1,22):
    print(i)
    
    f = F['X%s'%(i)]
    ls_sr.append(f.mean()/f.std()*np.sqrt(12))
    
    f = F[F.columns[:(i)]]
    w = pinv(f.cov()+lambda_cov*np.eye(f.shape[1])).dot(f.mean()+lambda_mean*np.ones(f.shape[1]))    
    w = w/np.sum(np.abs(w))
    mve = f.dot(w)
    ls_mve_sr.append(mve.mean()/mve.std()*np.sqrt(12))
    ls_mve_mean.append(mve.mean()*100)
    ls_mve_std.append(mve.std()*100)


df = pd.DataFrame([ ls_sr, ls_mve_sr, ls_mve_mean, ls_mve_std]).T
df.index = range(1,len(df.index)+1)
df.columns = ["SR","MVE SR","MVE Mean %","MVE Std %"]
df.to_csv("./tmp/sr_f.csv")

# Alpha


FF = pd.read_csv("../../../data//FactorsMonthly_202312.csv", index_col=0) /100
FF.index = pd.date_range(start="19630731",end="20220531",freq='M')

FF = FF[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
FF.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']

FF = FF.loc[start:end]
CAPM = FF['MKT']

FF=FF.values
CAPM=CAPM.values


l_a = [np.nan]
l_t = [np.nan]
l_r2 = [np.nan]
l_rmse = [np.nan]


for i in range(2,22):
    print(i)
    X = sm.add_constant(F[F.columns[:(i-1)]])
    Y = F['X%s'%(i)]

    model = sm.OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})

    l_a.append(results.params['const']*100)
    l_t.append(results.tvalues['const'])
    l_r2.append(results.rsquared)
    l_rmse.append(np.sqrt(np.mean(results.resid**2))*100)

l_ff5_a = []
l_ff5_t = []
l_ff5_r2 = []
l_ff5_rmse = []

for i in range(1,22):
    print(i)
    X = sm.add_constant(FF)
    Y = F['X%s'%(i)]

    model = sm.OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})

    l_ff5_a.append(results.params['const']*100)
    l_ff5_t.append(results.tvalues['const'])
    l_ff5_r2.append(results.rsquared)
    l_ff5_rmse.append(np.sqrt(np.mean(results.resid**2))*100)


l_capm_a = []
l_capm_t = []
l_capm_r2 = []
l_capm_rmse = []

# i = 1
for i in range(1,22):
    print(i)
    X = sm.add_constant(CAPM)
    Y = F['X%s'%(i)]

    model = sm.OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})

    l_capm_a.append(results.params['const']*100)
    l_capm_t.append(results.tvalues['const'])
    l_capm_r2.append(results.rsquared)
    l_capm_rmse.append(np.sqrt(np.mean(results.resid**2))*100)


df = pd.DataFrame([
                    l_capm_a,l_capm_t,l_capm_r2,
                   
                   l_ff5_a,l_ff5_t,l_ff5_r2,
                   
                    l_a,l_t,l_r2,
                   
                  ]
                 ).T
df.index = range(1,len(df.index)+1)
df.columns = [
                'CAPM-Alpha','CAPM-t-stat','CAPM-R2',
              
                'FF5-Alpha','FF5-t-stat','FF5-R2',
              
                'MF-Alpha','MF-t-stat','MF-R2',
              
             ]
df.to_csv('./tmp/factor_alpha_f.csv')


# merge two paty


tab1 = pd.read_csv("./tmp/sr_f.csv", index_col=0)
tab2 = pd.read_csv("./tmp/factor_alpha_f.csv", index_col=0)




tab = tab1.T.append(tab2.T).T
tab.to_csv("tabi6_f.csv")

