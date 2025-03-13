#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.api import OLS
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from numpy.linalg import pinv

# read P-Tree data


case='yr_2001_2020__1981_2000_num_iter_9_boost_20' 
folder = "P-Tree-c"

start = '2001-01-01' 
split = '2020-12-31' 
end = '2020-12-31' 

start = pd.to_datetime(start)
split = pd.to_datetime(split)
end = pd.to_datetime(end)

folder = "../"+folder+"/"
port_ins = pd.read_csv(folder+case+"_portfolio_fit1.csv", index_col=0)
weight = pd.read_csv(folder+case+"_leaf_weight1.csv", index_col=0)

# read factor data

FF = pd.read_csv("../../data/FactorsMonthly_202312.csv", index_col=0) /100
FF.index = pd.date_range(start="19630731",end="20220531",freq='M')
RF = FF['RF']

FF = FF.loc[start:end]
FF_ins = FF.loc[start:split][['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
CAPM_ins = FF_ins[['Mkt-RF']]

Q5 = pd.read_csv("../../data/Q5/q5_factors_monthly_2021.csv")
Q5.index = pd.date_range('1967-01-31', '2021-12-31', freq='M')
Q5.columns = ['year','month','rf','Q4MKT','Q4ME','Q4IA','Q4ROE','Q4EG']
Q5 = Q5[['Q4MKT','Q4ME','Q4IA','Q4ROE','Q4EG']]/100
Q5 = Q5.loc[start:end]
Q5_ins = Q5.loc[start:split]

RP = pd.read_csv("../../data/RP-PCA/rppca_ins_2001_2020.csv", header=None)
RP.index = pd.date_range(start.strftime("%Y-%m-%d"), split.strftime("%Y-%m-%d"), freq='M')
RP_ins = RP.loc[start.strftime("%Y"):split.strftime("%Y")]
print(RP_ins)

IP = pd.read_csv("../../data/IPCA/ipca_ins_2001_2020.csv", index_col=0)
IP.index = pd.date_range(start.strftime("%Y-%m-%d"), split.strftime("%Y-%m-%d"), freq='M')
IP_ins = IP.loc[start.strftime("%Y"):split.strftime("%Y")]
print(IP_ins)

# calculate alpha

list_sr=[]
list_avg=[]
list_avgt=[]
list_std=[]
list_alpha_CAPM=[]
list_alphat_CAPM=[]
list_beta_CAPM=[]
list_betat_CAPM=[]
list_R2_CAPM=[]
list_alpha_FF=[]
list_alphat_FF=[]
list_R2_FF=[]
list_alpha_Q5=[]
list_alphat_Q5=[]
list_R2_Q5=[]
list_alpha_RP=[]
list_alphat_RP=[]
list_R2_RP=[]
list_alpha_IP=[]
list_alphat_IP=[]
list_R2_IP=[]

for i in tqdm(port_ins.columns):
    Y = port_ins[i].values

    X = sm.add_constant(CAPM_ins)
    model = OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    list_alpha_CAPM.append(results.params['const']*100)
    list_alphat_CAPM.append(results.tvalues['const'])
    list_beta_CAPM.append(results.params['Mkt-RF'])
    list_betat_CAPM.append(results.tvalues['Mkt-RF'])
    list_R2_CAPM.append(results.rsquared)
    
    X = sm.add_constant(FF_ins)
    model = OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    list_alpha_FF.append(results.params['const']*100)
    list_alphat_FF.append(results.tvalues['const'])
    list_R2_FF.append(results.rsquared)

    X = sm.add_constant(Q5_ins)
    model = OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    list_alpha_Q5.append(results.params['const']*100)
    list_alphat_Q5.append(results.tvalues['const'])
    list_R2_Q5.append(results.rsquared)

    X = sm.add_constant(RP_ins)
    model = OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    list_alpha_RP.append(results.params['const']*100)
    list_alphat_RP.append(results.tvalues['const'])
    list_R2_RP.append(results.rsquared)

    X = sm.add_constant(IP_ins)
    model = OLS(Y,X)
    results = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
    list_alpha_IP.append(results.params['const']*100)
    list_alphat_IP.append(results.tvalues['const'])
    list_R2_IP.append(results.rsquared)
    list_sr.append(Y.mean()/Y.std()*np.sqrt(12))
    list_avg.append(Y.mean()*100)
    list_avgt.append(Y.mean()/Y.std()*np.sqrt(len(Y)))
    list_std.append(Y.std()*100)


tab1 = pd.DataFrame([
                     
                     list_avg,
                     list_std,
                     list_alpha_CAPM,
                     list_beta_CAPM,
                     list_R2_CAPM,
                     
                     list_alpha_FF,
                     list_alpha_Q5,
                     list_alpha_RP,
                     list_alpha_IP,
                    
                     list_R2_FF,
                     list_R2_Q5,
                     list_R2_RP,
                     list_R2_IP
                    ]).T
tab1.columns = [
    
                'AVG', 'STD',
                'A_CAPM','B_CAPM','R2_CAPM',
                'A_FF5','A_Q5','A_RP5','A_IP5',
               'R2_FF5','R2_Q5','R2_RP5','R2_IP5'
               ]
tab1 = tab1.round(decimals=2)
tab1.to_csv("tmp/coef_c.csv")

# t value
tab2 = pd.DataFrame([
                     list_avgt,
                     np.zeros(len(list_alphat_FF))*np.nan,
                     
                     list_alphat_CAPM,
                     list_betat_CAPM,
                     np.zeros(len(list_alphat_FF))*np.nan,
                     
                     list_alphat_FF,
                     list_alphat_Q5,
                     list_alphat_RP,
                     list_alphat_IP,
                    
                     np.zeros(len(list_alphat_FF))*np.nan,
                     np.zeros(len(list_alphat_FF))*np.nan,
                     np.zeros(len(list_alphat_FF))*np.nan,
                     np.zeros(len(list_alphat_FF))*np.nan,
                    ]).T
tab2.columns = [
                'AVG','STD',
                'A_CAPM','B_CAPM','R2_CAPM',
                'A_FF5','A_Q5','A_RP5','A_IP5',
               'R2_FF5','R2_Q5','R2_RP5','R2_IP5'
               ]
tab2 = tab2.round(decimals=2)
tab2.to_csv("./tmp/t_c.csv")

tab3 = pd.DataFrame([])
for i in tab1.columns:
    tmp1 = tab1[i]
    tmp2 = tab2[i]
    
    tmp3 = []
    for j in range(len(tmp1)):
        tmp3.append(tmp1[j])
        if np.isnan(tmp2[j]):
            tmp3.append("$ $")
        else:
            tmp3.append("[%s]"%tmp2[j])
    
    tab3[i] = tmp3

tmp_index = []
for i in range(1,len(tab1.index)+1):
    tmp_index.append(i)
    tmp_index.append("$ $")

tab3.index = tmp_index
tab3.to_csv("tab1_c.csv")
tab3

