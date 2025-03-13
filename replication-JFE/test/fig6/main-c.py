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
import warnings
warnings.filterwarnings('ignore') 

case0 = 'MEBM25'
case = "P-Tree-c"

start = '2001-01-31'
split = '2020-12-31'
end   = '2020-12-31'


# read data

FF = pd.read_csv("../../data/FactorsMonthly_202312.csv", index_col=0) /100
FF.index = pd.date_range(start="19630731",end="20220531",freq='M')
RF = FF['RF']
FF = FF.loc[start:end]
FF_ins = FF.loc[start:split][['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
CAPM_ins = FF_ins[['Mkt-RF']]

MEBM25 = pd.read_csv("../../data/download_portfolios/sort25/25_ME_BM.CSV", index_col=0) /100
MEBM25.index = pd.date_range(start="19260731",end="20230430",freq='M')
MEBM25 = MEBM25.loc[start:end]

# calculate mean, std, alpha, beta of BEME25

port_ins = MEBM25
list_sr=[]
list_avg=[]
list_avgt=[]
list_std=[]
list_alpha_CAPM=[]
list_alphat_CAPM=[]
list_beta_CAPM=[]
list_betat_CAPM=[]
list_R2_CAPM=[]


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
    list_sr.append(Y.mean()/Y.std()*np.sqrt(12))
    list_avg.append(Y.mean()*100)
    list_avgt.append(Y.mean()/Y.std()*np.sqrt(len(Y)))
    list_std.append(Y.std()*100)


tab1 = pd.DataFrame([
                     list_sr,
                     
                     list_avg,
                     list_std,
                     list_alpha_CAPM,
                     list_beta_CAPM,
                     list_R2_CAPM,
                     
                    ]).T
tab1.columns = [
                'SR',
                'AVG', 'STD',
                'A_CAPM','B_CAPM','R2_CAPM'
               ]
tab1 = tab1.round(decimals=2)
tab1.to_csv("./tmp/asset-tab1-"+case0+"-ins.csv")


# plot P-Tree Test Assets and BEME25 in one figure

tab1_pt = pd.read_csv("../tab1/tmp/coef_c.csv", index_col=0)

list_avg_pt = tab1_pt['AVG']
list_std_pt = tab1_pt['STD']
plt.figure(figsize=(5,5))
plt.xlim(-4,4)
plt.ylim(3,11)
plt.scatter(list_avg_pt, list_std_pt, color = 'black', alpha=1, marker='o')
plt.scatter(list_avg, list_std, color = 'red', alpha=0.25, marker='^')
plt.xlabel("Mean %")
plt.ylabel("Std %")
plt.savefig("mean-std-2-"+case+"+"+case0+".pdf", bbox_inches='tight')
plt.close()


list_alpha_CAPM_pt = tab1_pt['A_CAPM']
list_beta_CAPM_pt = tab1_pt['B_CAPM']
plt.figure(figsize=(5,5))
plt.xlim(-5,5)
plt.ylim(0.4,1.6)
plt.scatter(list_alpha_CAPM_pt, list_beta_CAPM_pt, color = 'black', alpha=1, marker='o')
plt.scatter(list_alpha_CAPM, list_beta_CAPM, color = 'red', alpha=0.25, marker='^')
plt.xlabel("Alpha %")
plt.ylabel("Beta")
plt.savefig("alpha-beta-2-"+case+"+"+case0+".pdf", bbox_inches='tight')
plt.close()

