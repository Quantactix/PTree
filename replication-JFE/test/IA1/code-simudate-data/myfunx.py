
BB = 11
# case_simu = "Oct"

import pandas as pd
import matplotlib.pyplot as plt
import os
import pyarrow.feather as fr
from tqdm import tqdm
import numpy as np
import random

K = 61
N = 1000
T = 1000

names = ['ill', 'me_ia', 'chtx', 'mom36m', 're', 'depr', 'rd_sale', 'roa',
    'bm_ia', 'cfp', 'mom1m', 'baspread', 'rdm', 'bm', 'sgr', 'mom12m',
    'std_dolvol', 'rvar_ff3', 'herf', 'sp', 'hire', 'pctacc',
    'grltnoa', 'turn', 'abr', 'seas1a', 'adm', 'me', 'cash', 'chpm',
    'cinvest', 'acc', 'gma', 'beta', 'sue', 'cashdebt', 'ep', 'lev',
    'op', 'alm', 'lgr', 'noa', 'roe', 'dolvol', 'rsup', 'std_turn',
    'maxret', 'mom6m', 'ni', 'nincr', 'ato', 'rna', 'agr', 'zerotrade',
    'chcsho', 'dy', 'rvar_capm', 'svar', 'mom60m', 'pscore', 'pm']
len(names)


Sigma = pd.read_csv("../data/calibration_Sigma.csv", index_col=0)
Coef = pd.read_csv("../data/calibration_coef.csv", index_col=0)
A = Coef.iloc[0,:]
B = Coef.iloc[1:,:]

dgp_coef = pd.read_csv("../data/dgp_coef.csv", index_col=0)['0']
parameter_mkt = pd.read_csv("../data/parameter_mkt.csv", index_col=0)['0']


def myDGP(kappa, seq):

    random.seed(seq)
    np.random.seed(seq)

    myseed = int(np.random.uniform(0,1000000))
    print(myseed)
    random.seed(myseed)
    np.random.seed(myseed)

    # simulation MKTRF
    mkt_simulation = np.random.normal(parameter_mkt['mean'], parameter_mkt['std'], T)
    mkt_simulation = pd.Series(mkt_simulation)

    df_mktrf = pd.DataFrame()
    df_mktrf['date'] = range(T)
    df_mktrf['mktrf'] = mkt_simulation

    # simulation xxret
    da_simu = pd.DataFrame()

    init_char = np.random.uniform(0,1,(N,K))
    print(init_char.shape)
    tmp = pd.DataFrame(init_char)
    tmp['permno'] = range(N)
    d = 0
    tmp['date'] = d
    da_simu = da_simu.append(tmp)

    for i in range(T-1):
        
        next_char = init_char.dot(B) + np.diag(np.ones(K)).dot(A) + np.random.multivariate_normal(np.zeros(K), Sigma, N)
        init_char = next_char
        tmp = pd.DataFrame(init_char)
        tmp['permno'] = range(N)
        d = d+1
        tmp['date'] = d
        da_simu = da_simu.append(tmp)

    da_simu.columns = names + ['permno', 'date']

    # Do rank and standardize to [-1, 1]
    newframe = []
    datelist = list(da_simu['date'].unique())
    for month in datelist:
        tmp_da_simu = da_simu[da_simu['date'] == month].reset_index(drop=True)
        for char in names:
            tmp_da_simu[char] = tmp_da_simu[char].rank(method='dense')
            tmp_da_simu[char] = (tmp_da_simu[char] - np.min(tmp_da_simu[char])) / (np.max(tmp_da_simu[char]) - np.min(tmp_da_simu[char])) * 2 -1
        newframe.append(tmp_da_simu)
    da_simu = pd.concat(newframe, axis=0).reset_index(drop=True)

    da4 = da_simu[['mom12m','bm','me','date']]
    da4['mom12m_square'] = da4['mom12m']**2
    da4['me_bm'] = da4['me']*da4['bm']

    # cross-sectional de-mean the regressors.
    newframe = []
    datelist = list(da4['date'].unique())
    for month in datelist:
        tmp_da4 = da4[da4['date'] == month].reset_index(drop=True)
        for char in ['me', 'bm', 'me_bm','mom12m', 'mom12m_square']:
            tmp_da4[char] = tmp_da4[char] - np.mean(tmp_da4[char])
        newframe.append(tmp_da4)
    da4 = pd.concat(newframe, axis=0).reset_index(drop=True)
    da4['xxret-sig-raw'] = dgp_coef[1] * da4['me'] + dgp_coef[2] * da4['bm'] + dgp_coef[3] * da4['me_bm'] + dgp_coef[4] * da4['mom12m'] + dgp_coef[5] * da4['mom12m_square']
    
    # merge market
    da5 = da4.merge(df_mktrf, how='left', on='date')
    sigma = dgp_coef[6]
    noise = np.random.normal(0, sigma, N*T) 

    vs = da4['xxret-sig-raw'].var()
    print("START: kappa: \n")
    print(kappa)
    print("END: kappa: \n")

    da5['xxret-sig'] = kappa * da5['xxret-sig-raw']
    da5['xxret'] = da5['xxret-sig'] + noise
    da5['xret'] = da5['xxret'] + da5['mktrf']    

    var_tot = da5['xxret'].var()
    var_sig = da5['xxret-sig'].var()
    var_nos = noise.var()

    print( "Check signal to total:%s"%(var_sig/var_tot) )
    print( "Check noise  to total:%s"%(var_nos/var_tot) )

    da_out = da_simu.copy()
    da_out = da_out.reset_index()

    da_out['xret'] = da5['xret']
    da_out['xxret-sig'] = da5['xxret-sig']
    da_out['xxret'] = da5['xxret']
    da_out['noise'] = noise
    da_out['mktrf'] = da5['mktrf']
    da_out['mom12m_square'] = da5['mom12m_square']
    da_out['me_bm'] = da5['me_bm']

    da_out = da_out[[
        'permno', 'date', 
        'xret',  'xxret', 'xxret-sig', 'noise', 'mktrf',

        'mom12m_square', 'bm','me', 'me_bm', 'mom12m',
        
        'ill', 'me_ia', 'chtx', 'mom36m', 're', 'depr', 'rd_sale', 'roa',
        'bm_ia', 'cfp', 'mom1m', 'baspread', 'rdm',  'sgr', 
        'std_dolvol', 'rvar_ff3', 'herf', 'sp', 'hire', 'pctacc',
        'grltnoa', 'turn', 'abr', 'seas1a', 'adm', 'cash', 'chpm',
        'cinvest', 'acc', 'gma', 'beta', 'sue', 'cashdebt', 'ep', 'lev',
        'op', 'alm', 'lgr', 'noa', 'roe', 'dolvol', 'rsup', 'std_turn',
        'maxret', 'mom6m', 'ni', 'nincr', 'ato', 'rna', 'agr', 'zerotrade',
        'chcsho', 'dy', 'rvar_capm', 'svar', 'mom60m', 'pscore', 'pm'
    ]]

    fr.write_feather(da_out,"../data/simu_kappa_%s_seed_%s.feather"%(kappa,seq))

    print("##### SNR %s; SEQ %s; FINISHED #####")
