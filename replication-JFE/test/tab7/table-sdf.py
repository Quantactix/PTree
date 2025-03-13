#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from numpy.linalg import inv
import pyarrow.feather as fr

def make_one_folder(xt,case,in_or_out):
    
    if case != 'all':
        folder = './%s/%s'%(xt,case)    
        f = pd.read_csv(folder+"/G-m2_bf_"+in_or_out+".csv", index_col=0)[['X1','X2']]
        print(f.head())

        w = inv(f.cov()).dot(f.mean())
        w = w/w.sum()
        mve = f.dot(w)

        avg = mve.mean()*100
        sr = mve.mean()/mve.std()*np.sqrt(12)

        df = pd.DataFrame([avg,sr]).T
        df.columns = ['AVG','SR']
        df.index = [5]
        return df
    else:
        mvea = []
        for case in ['top','btm']:
            print(case)
            folder = './%s/%s'%(xt,case)    
            f = pd.read_csv(folder+"/G-m2_bf_"+in_or_out+".csv", index_col=0)[['X1','X2']]

            w = inv(f.cov()).dot(f.mean())
            w = w/w.sum()
            mve = f.dot(w)

            mvea = mvea + list(mve)

        avg = np.mean(mvea)*100
        sr = np.mean(mvea)/np.std(mvea)*np.sqrt(12)

        df = pd.DataFrame([avg,sr]).T
        df.columns = ['AVG','SR']
        df.index = [5]
        return df

xt_list = ['x_infl','x_dy', 'x_lev', 'x_ep', 'x_ni',
            'x_tbl', 'x_dfy', 'x_tms', 'x_svar', 'x_ill']

xt_list.sort()

case_list = ['top','btm','all']

df_xt_all = pd.DataFrame([])
for xt in xt_list:
    print(xt)
    df_xt = pd.DataFrame([])
    list_xt_result = []
    for case in case_list:
        df = make_one_folder(xt,case,'train')
        tmp = df.loc[5]
        df_xt = df_xt.append(tmp)
    df_xt.index = [ xt+"-"+i for i in case_list ]
    df_xt_all = df_xt_all.append(df_xt)

df_xt_all.to_csv("tab-invest-xt-all-PT1.csv")
