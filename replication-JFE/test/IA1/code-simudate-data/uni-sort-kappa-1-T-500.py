from myfunx import *
kappa = 1


T_split = 500
lambda_mean = 0 
lambda_cov  = 1e-4

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pyarrow.feather as fr
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from numpy.linalg import inv, pinv

names = [    
    'mom12m', 'me', 'bm', 
    
    'ill', 'me_ia', 'chtx', 'mom36m', 're', 'depr', 'rd_sale', 'roa',
    'bm_ia', 'cfp', 'mom1m', 'baspread', 'rdm', 'sgr', 
    'std_dolvol', 'rvar_ff3', 'herf', 'sp', 'hire', 'pctacc',
    'grltnoa', 'turn', 'abr', 'seas1a', 'adm', 'cash', 'chpm',
    'cinvest', 'acc', 'gma', 'beta', 'sue', 'cashdebt', 'ep', 'lev',
    'op', 'alm', 'lgr', 'noa', 'roe', 'dolvol', 'rsup', 'std_turn',
    'maxret', 'mom6m', 'ni', 'nincr', 
    'ato', 'rna', 'agr', 'zerotrade',
    'chcsho', 'dy', 'rvar_capm', 'svar', 'mom60m', 'pscore', 'pm']
len(names)


def getAlpha(mkt,asset):
    # Alpha
    # DA = pd.DataFrame([mkt.values,asset.values]).T;DA.columns = ['MKT','asset']
    DA = pd.DataFrame([mkt,asset]).T;DA.columns = ['MKT','asset']
    X = sm.add_constant(DA[['MKT']]);Y = DA[['asset']];
    model = sm.OLS(Y,X);results = model.fit()
    Alpha = results.params[0]
    return Alpha

def getAvgAlpha(mkt,df):
    list_alpha = []
    
    mkt.index = range(len(mkt.index))
    df.index = range(len(df.index))

    for i in df.columns:
        print(i)
        asset = df[i]
        list_alpha.append(getAlpha(mkt,asset))
    list_abs = [np.abs(i) for i in list_alpha]
    list_sqr = [i**2 for i in list_alpha]
    return np.mean(list_abs)*100, 100*np.mean(list_sqr)**0.5

def mve_of_asset(port, df_ins, df_oos):
    ts_ins = da_ins.groupby(['date',port]).mean()['xret']
    ts_ins = ts_ins.unstack()
    ts_ins.index = range(len(ts_ins.index))
    ts_ins.columns = [i+str(j) for j in ts_ins.columns]
    df_ins = df_ins.T.append(ts_ins.T).T

    absAlp_ins, sqrAlp_ins = getAvgAlpha(mkt_ins,ts_ins)

    ts_oos = da_oos.groupby(['date',port]).mean()['xret']
    ts_oos = ts_oos.unstack()
    ts_oos.index = range(len(ts_oos.index))
    ts_oos.columns = [i+str(j) for j in ts_oos.columns]
    df_oos = df_oos.T.append(ts_oos.T).T
    
    absAlp_oos, sqrAlp_oos = getAvgAlpha(mkt_oos,ts_oos)
    
    sigma = ts_ins.cov()
    mu = ts_ins.mean()
    n = ts_ins.shape[1]
    w = pinv(sigma+np.diag(np.ones(n))*lambda_cov).dot(mu+lambda_mean)
    w = w/sum(np.abs(w))

    sdf_ins = ts_ins.dot(w)
    sr_ins = sdf_ins.mean()/sdf_ins.std()*np.sqrt(12)

    sdf_oos = ts_oos.dot(w)
    sr_oos = sdf_oos.mean()/sdf_oos.std()*np.sqrt(12)
    
    return (sr_ins, sr_oos, df_ins, df_oos, sdf_ins, sdf_oos, absAlp_ins, sqrAlp_ins, absAlp_oos, sqrAlp_oos)

df_Sharpe_ins = pd.DataFrame()
df_Sharpe_oos = pd.DataFrame()

df_aa_ins = pd.DataFrame()
df_aa_oos = pd.DataFrame()

df_sa_ins = pd.DataFrame()
df_sa_oos = pd.DataFrame()

# for seq in tqdm(range(1,11)):
for seq in tqdm(range(1,BB)):

    da = fr.read_feather("../data/simu_kappa_%s_seed_%s.feather"%(kappa,seq))
    del da['mom12m_square'], da['me_bm']
    # da = standardize(da)

    mkt = da.groupby('date').first()['mktrf']
    mkt_ins = mkt.loc[:(T_split-1)]
    mkt_oos = mkt.loc[T_split:]

    for i in names:
        da['uni10-%s'%i] = np.where(da[i]>=-0.8,1,0)\
                        +np.where(da[i]>=-0.6,1,0)\
                        +np.where(da[i]>=-0.4,1,0)\
                        +np.where(da[i]>=-0.2,1,0)\
                        +np.where(da[i]>=0,1,0)\
                        +np.where(da[i]>=0.2,1,0)\
                        +np.where(da[i]>=0.4,1,0)\
                        +np.where(da[i]>=0.6,1,0)\
                        +np.where(da[i]>=0.8,1,0)\
                        +1

    for i in ['me','bm','mom12m']:
        da['uni5-%s'%i] = np.where(da[i]>=-0.6,1,0)\
                        + np.where(da[i]>=-0.2,1,0)\
                        + np.where(da[i]>=0.2,1,0)\
                        + np.where(da[i]>=0.6,1,0)\
                        + 1
    for i in ['me','bm','mom12m']:
        da['uni3-%s'%i] = np.where(da[i]>=-0.333,1,0)\
                        + np.where(da[i]>=0.333,1,0)\
                        + 1
        
    # size-value 5x5
    da['me5-bm5'] = ["%s-%s"%(da['uni5-me'][i],da['uni5-bm'][i]) for i in da.index]

    # size-momemtum 5x5
    da['me5-mom12m5'] = ["%s-%s"%(da['uni5-me'][i],da['uni5-mom12m'][i]) for i in da.index]

    # value-momentum 5x5
    da['bm5-mom12m5'] = ["%s-%s"%(da['uni5-bm'][i],da['uni5-mom12m'][i]) for i in da.index]

    da_ins = da[da['date']<T_split]
    da_oos = da[da['date']>=T_split]

    print(da_ins.shape)
    print(da_oos.shape)

    # start calculate SDF
    df_ins = pd.DataFrame()
    df_oos = pd.DataFrame()

    Series_Sharpe_ins = pd.Series()
    Series_Sharpe_oos = pd.Series()

    Series_aa_ins = pd.Series()
    Series_sa_ins = pd.Series()

    Series_aa_oos = pd.Series()
    Series_sa_oos = pd.Series()

    # top 3 char
    for i in names[:3]:
        (sr_ins, sr_oos, df_ins, df_oos, sdf_ins, sdf_oos, absAlp_ins, sqrAlp_ins, absAlp_oos, sqrAlp_oos) = mve_of_asset("uni10-%s"%i, df_ins, df_oos)
        
        Series_Sharpe_ins[i] = sr_ins
        Series_Sharpe_oos[i] = sr_oos
        
        Series_aa_ins[i] = absAlp_ins
        Series_sa_ins[i] = sqrAlp_ins

        Series_aa_oos[i] = absAlp_oos
        Series_sa_oos[i] = sqrAlp_oos

        sdf_ins.to_csv("../data/tmp-reg/sdf_ins_%s_%s_%s.csv"%(kappa,i,seq))
        sdf_oos.to_csv("../data/tmp-reg/sdf_oos_%s_%s_%s.csv"%(kappa,i,seq))

    # 30 sdf
    sigma = df_ins.cov()
    mu = df_ins.mean()

    n = df_ins.shape[1]
    w = pinv(sigma+np.diag(np.ones(n))*lambda_cov).dot(mu+lambda_mean)
    w = w/sum(np.abs(w))

    sdf_ins = df_ins.dot(w)
    sdf_oos = df_oos.dot(w)

    if sdf_ins.mean()<0:
        sdf_ins = -1*sdf_ins
        sdf_oos = -1*sdf_oos

    sr_ins = sdf_ins.mean()/sdf_ins.std()*np.sqrt(12)
    sr_oos = sdf_oos.mean()/sdf_oos.std()*np.sqrt(12)

    Series_Sharpe_ins['sdf-30'] = sr_ins
    Series_Sharpe_oos['sdf-30'] = sr_oos

    sdf_ins.to_csv("../data/tmp-reg/sdf_ins_%s_%s_%s.csv"%(kappa,"sdf-30",seq))
    sdf_oos.to_csv("../data/tmp-reg/sdf_oos_%s_%s_%s.csv"%(kappa,"sdf-30",seq))

    absAlp_ins, sqrAlp_ins = getAvgAlpha(mkt_ins, df_ins)
    absAlp_oos, sqrAlp_oos = getAvgAlpha(mkt_oos, df_oos)

    Series_aa_ins['sdf-30'] = absAlp_ins
    Series_sa_ins['sdf-30'] = sqrAlp_ins

    Series_aa_oos['sdf-30'] = absAlp_oos
    Series_sa_oos['sdf-30'] = sqrAlp_oos

    # other char
    for i in names[3:]:
        (sr_ins, sr_oos, df_ins, df_oos, sdf_ins, sdf_oos, absAlp_ins, sqrAlp_ins, absAlp_oos, sqrAlp_oos) = mve_of_asset("uni10-%s"%i, df_ins, df_oos)
    
        Series_Sharpe_ins[i] = sr_ins
        Series_Sharpe_oos[i] = sr_oos

        Series_aa_ins[i] = absAlp_ins
        Series_sa_ins[i] = sqrAlp_ins

        Series_aa_oos[i] = absAlp_oos
        Series_sa_oos[i] = sqrAlp_oos
        
        sdf_ins.to_csv("../data/tmp-reg/sdf_ins_%s_%s.csv"%(kappa,i))
        sdf_oos.to_csv("../data/tmp-reg/sdf_oos_%s_%s.csv"%(kappa,i))

    # 610 sdf
    sigma = df_ins.cov()
    mu = df_ins.mean()
    
    n = df_ins.shape[1]
    w = pinv(sigma+np.diag(np.ones(n))*lambda_cov).dot(mu+lambda_mean)
    w = w/sum(np.abs(w))
    
    sdf_ins = df_ins.dot(w)
    sdf_oos = df_oos.dot(w)

    if sdf_ins.mean()<0:
        sdf_ins = -1*sdf_ins
        sdf_oos = -1*sdf_oos

    sr_ins = sdf_ins.mean()/sdf_ins.std()*np.sqrt(12)
    sr_oos = sdf_oos.mean()/sdf_oos.std()*np.sqrt(12)

    Series_Sharpe_ins['sdf-610'] = sr_ins
    Series_Sharpe_oos['sdf-610'] = sr_oos

    sdf_ins.to_csv("../data/tmp-reg/sdf_ins_%s_%s_%s.csv"%(kappa,"sdf-610",seq))
    sdf_oos.to_csv("../data/tmp-reg/sdf_oos_%s_%s_%s.csv"%(kappa,"sdf-610",seq))

    absAlp_ins, sqrAlp_ins = getAvgAlpha(mkt_ins, df_ins)
    absAlp_oos, sqrAlp_oos = getAvgAlpha(mkt_oos, df_oos)

    Series_aa_ins['sdf-610'] = absAlp_ins
    Series_sa_ins['sdf-610'] = sqrAlp_ins

    Series_aa_oos['sdf-610'] = absAlp_oos
    Series_sa_oos['sdf-610'] = sqrAlp_oos

    # size value
    i = "me5-bm5"
    (sr_ins, sr_oos, df_ins, df_oos, sdf_ins, sdf_oos, absAlp_ins, sqrAlp_ins, absAlp_oos, sqrAlp_oos) = mve_of_asset(i, df_ins, df_oos)
    Series_Sharpe_ins[i] = sr_ins
    Series_Sharpe_oos[i] = sr_oos
    Series_aa_ins[i] = absAlp_ins
    Series_sa_ins[i] = sqrAlp_ins
    Series_aa_oos[i] = absAlp_oos
    Series_sa_oos[i] = sqrAlp_oos

    sdf_ins.to_csv("../data/tmp-reg/sdf_ins_%s_%s_%s.csv"%(kappa,i,seq))
    sdf_oos.to_csv("../data/tmp-reg/sdf_oos_%s_%s_%s.csv"%(kappa,i,seq))

    # size mom12m
    i = "me5-mom12m5"
    (sr_ins, sr_oos, df_ins, df_oos, sdf_ins, sdf_oos, absAlp_ins, sqrAlp_ins, absAlp_oos, sqrAlp_oos) = mve_of_asset(i, df_ins, df_oos)
    Series_Sharpe_ins[i] = sr_ins
    Series_Sharpe_oos[i] = sr_oos
    Series_aa_ins[i] = absAlp_ins
    Series_sa_ins[i] = sqrAlp_ins
    Series_aa_oos[i] = absAlp_oos
    Series_sa_oos[i] = sqrAlp_oos

    sdf_ins.to_csv("../data/tmp-reg/sdf_ins_%s_%s_%s.csv"%(kappa,i,seq))
    sdf_oos.to_csv("../data/tmp-reg/sdf_oos_%s_%s_%s.csv"%(kappa,i,seq))

    # value mom12m
    i = "bm5-mom12m5"
    (sr_ins, sr_oos, df_ins, df_oos, sdf_ins, sdf_oos, absAlp_ins, sqrAlp_ins, absAlp_oos, sqrAlp_oos) = mve_of_asset(i, df_ins, df_oos)
    Series_Sharpe_ins[i] = sr_ins
    Series_Sharpe_oos[i] = sr_oos
    Series_aa_ins[i] = absAlp_ins
    Series_sa_ins[i] = sqrAlp_ins
    Series_aa_oos[i] = absAlp_oos
    Series_sa_oos[i] = sqrAlp_oos

    sdf_ins.to_csv("../data/tmp-reg/sdf_ins_%s_%s_%s.csv"%(kappa,i,seq))
    sdf_oos.to_csv("../data/tmp-reg/sdf_oos_%s_%s_%s.csv"%(kappa,i,seq))

    df_Sharpe_ins[seq] = Series_Sharpe_ins
    df_Sharpe_oos[seq] = Series_Sharpe_oos

    df_aa_ins[seq] = Series_aa_ins
    df_aa_oos[seq] = Series_aa_oos
    df_sa_ins[seq] = Series_sa_ins
    df_sa_oos[seq] = Series_sa_oos

df_Sharpe_ins['avg'] = df_Sharpe_ins.mean(1)
df_Sharpe_ins.to_csv("../data/df_SharpeAlpha_ins_%s.csv"%(kappa))
df_Sharpe_oos['avg'] = df_Sharpe_oos.mean(1)
df_Sharpe_oos.to_csv("../data/df_SharpeAlpha_oos_%s.csv"%(kappa))

df_aa_ins['avg'] = df_aa_ins.mean(1)
df_aa_ins.to_csv("../data/df_aaAlpha_ins_%ss.csv"%(kappa))
df_aa_oos['avg'] = df_aa_oos.mean(1)
df_aa_oos.to_csv("../data/df_aaAlpha_oos_%s.csv"%(kappa))

df_sa_ins['avg'] = df_sa_ins.mean(1)
df_sa_ins.to_csv("../data/df_saAlpha_ins_%s.csv"%(kappa))
df_sa_oos['avg'] = df_sa_oos.mean(1)
df_sa_oos.to_csv("../data/df_saAlpha_oos_%s.csv"%(kappa))

out = pd.DataFrame(index = df_Sharpe_ins.index)

out['SR-ins'] = df_Sharpe_ins['avg'].values
out['AA-ins'] = df_aa_ins['avg'].values
out['SA-ins'] = df_sa_ins['avg'].values

out['SR-oos'] = df_Sharpe_oos['avg'].values
out['AA-oos'] = df_aa_oos['avg'].values
out['SA-oos'] = df_sa_oos['avg'].values

out.to_csv("df_out_%s.csv"%(kappa))