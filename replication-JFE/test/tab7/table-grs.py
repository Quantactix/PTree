#!/usr/bin/env python
# coding: utf-8

import pyarrow.feather as fr
import pandas as pd
import datetime as dt
from scipy.stats import f
from scipy import linalg
from numpy.linalg import pinv
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd 


# Regression
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
    # a = (T - N - K)/N
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
    
    return (GRS, pvalue, 
            100*Alpha.abs().mean(), 
            (10000*Alpha**2).mean(),
            100*(Alpha**2).mean()**0.5,
            rmse*100, R2.mean())

start = '1981-01-01'
end = '2020-12-31'

FF = pd.read_csv("../../data/FactorsMonthly_202205.csv", index_col=0) /100
FF.index = pd.date_range(start="19630731",end="20220531",freq='M')
FF['Const'] = 1

FF = FF.loc[start:end]
RF = FF['RF']
FF.index = range(len(FF.index))
FF = FF[['Const','Mkt-RF']]


def get_PTREE_asset_and_index(xt, tob):

    R = pd.read_csv("./x_"+xt[2:]+"/"+tob+"/_basis_portfolio_"+tob+"_.csv", index_col=0)
    idx = pd.read_csv("./x_"+xt[2:]+"/"+tob+"/G-m2_train_ts_idx.csv", index_col=0)['train_ts_idx']
    return R, idx

def make_one_folder(xt,in_or_out):
    R, idx = get_PTREE_asset_and_index(xt, in_or_out)
    capm = FF.iloc[FF.index[idx],:]; capm.index = range(1,1+len(capm.index))
    (grs,p,a,a2,ra2,rmse,r2) = myGRS_plus_rmse(capm,R)
    return grs,p,a,a2,ra2,rmse,r2
    

xt_list = ['x_infl','x_dy', 'x_lev', 'x_ep', 'x_ni',
            'x_tbl', 'x_dfy', 'x_tms', 'x_svar', 'x_ill']

xt_list.sort()

case_list = ['top','btm']

df = pd.DataFrame([])
for xt in xt_list:
    for case in case_list:
        print(xt)
        grs,p,a,a2,ra2,rmse,r2 = make_one_folder(xt,case)
        df[xt+"-"+case] = [grs,p,a,ra2,r2]

df.to_csv("tab-ts-GRS.csv")
