
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy.linalg import pinv


start = '1981-01-31'
split = '2020-12-31'
end   = '2020-12-31'

# P-Tree data

lambda_mean = 0
lambda_cov = 1e-5

case = "Gyr_1981_2020_num_iter_9_boost_20"
folder = "P-Tree-d"

folder = "../"+folder+"/"
Fins = pd.read_csv(folder+case+"_bf_train.csv", index_col=0)
Fins.head()


# Other factors

FF = pd.read_csv("../../../data/FactorsMonthly_202312.csv", index_col=0) /100
FF.index = pd.date_range(start="19630731",end="20220531",freq='M')

FF = FF[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
FF.columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']

FF = FF.loc[start:split]
CAPM = FF['MKT']

FF=FF.values
CAPM=CAPM.values

# q5
df_q5 = pd.read_csv("../../../data/Q5/q5_factors_monthly_2021.csv")/100
df_q5.index = pd.date_range(start="19670131",end="20211231",freq='M')
df_q5 = df_q5.loc[start:end]
df_q5 = df_q5[["R_MKT","R_ME","R_IA","R_ROE","R_EG"]]

Q5 = df_q5.values

# rp
df_rppca = pd.read_csv("../../../data/RP-PCA/rppca_1981_2020.csv", header=None)
RP = df_rppca.values

# ip
df_ipca = pd.read_csv("../../../data/IPCA/ipca_1981_2020.csv", index_col=0)
IP = df_ipca.values


# outputs
ls_mve_sr = []

l_a_capm = []
l_t_capm = []

l_a_ff5 = []
l_t_ff5 = []

l_a_q5 = []
l_t_q5 = []

l_a_rp5 = []
l_t_rp5 = []

l_a_ip5 = []
l_t_ip5 = []


for i in [1,5,10,15,20]:
    F = Fins[Fins.columns[:(i+1)]]
    print(F.head())

    ## MVE
    w = pinv(F.cov()+np.eye(F.shape[1])*lambda_cov).dot(F.mean())
    w = w/sum(w)

    mve = F.dot(w)
    sr1 = np.mean(mve)/np.std(mve)
    print(sr1)
    sr = np.mean(mve)/np.std(mve)*np.sqrt(12)
    print(sr)
    ls_mve_sr.append(sr)

    ## Alpha
    X = sm.add_constant(CAPM)
    Y = mve
    model = sm.OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    print(results.params['const']*100)
    print(results.tvalues['const'])
    l_a_capm.append(results.params['const']*100)
    l_t_capm.append(results.tvalues['const'])

    X = sm.add_constant(FF)
    Y = mve
    model = sm.OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    print(results.params['const']*100)
    print(results.tvalues['const'])
    l_a_ff5.append(results.params['const']*100)
    l_t_ff5.append(results.tvalues['const'])
    
    X = sm.add_constant(Q5)
    Y = mve
    model = sm.OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    print(results.params['const']*100)
    print(results.tvalues['const'])
    l_a_q5.append(results.params['const']*100)
    l_t_q5.append(results.tvalues['const'])
    
    X = sm.add_constant(RP)
    Y = mve
    model = sm.OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    print(results.params['const']*100)
    print(results.tvalues['const'])
    l_a_rp5.append(results.params['const']*100)
    l_t_rp5.append(results.tvalues['const'])
    
    X = sm.add_constant(IP)
    Y = mve
    model = sm.OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    print(results.params['const']*100)
    print(results.tvalues['const'])
    l_a_ip5.append(results.params['const']*100)
    l_t_ip5.append(results.tvalues['const'])

df_1 = pd.DataFrame([ls_mve_sr, l_a_capm, l_a_ff5, l_a_q5, l_a_rp5, l_a_ip5]).T
df_2 = pd.DataFrame([l_t_capm, l_t_ff5, l_t_q5, l_t_rp5, l_t_ip5]).T

df_1.to_csv('./tmp/tabi7-d-estimate.csv')
df_2.to_csv('./tmp/tabi7-d-tstat.csv')