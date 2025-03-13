#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from numpy.linalg import inv
from numpy import matmul
from tqdm import tqdm
from collections import Counter
from scipy import stats

B=1001

# read regression data
start = '1981-01-01'
split = '2020-12-31'
case = 'rf_vw_noh_1981'

import pyarrow.feather as fr
data = fr.read_feather('../data/data_202312.feather') # also works for Rds

# library(arrow)
# my_tibble = arrow::read_feather("../../data/data_202312.feather")
# data <- as.data.frame(my_tibble)

### START: split related variable importance ###
def find_start_end_chars(b, case):
    # b = 7
    file = "./"+case+"/log_%s.txt"%b
    f = open(file, "r")

    num_lines = sum(1 for line in f)
    
    f.close()
    ### find the start and end of tree structure
    f = open(file, "r")
    start=0
    end=0
    i = 0
    flag1 = False
    for x in f:
        if x[:2] == '1 ':
            print('#comments# node 1')
            flag1=True
            start = i
        if x == '\n' and flag1==True:
            print('#comments# end the nodes')
            end = i
            break
        i=i+1
    print('start of nodes: row %s'%start)
    print('end of nodes: row %s'%end)
    vnames = pd.read_csv('./'+case+'/output_%s/splitting_chars.csv'%b, index_col=0)['splitting_chars'].values
    return (start,end,vnames)

def get_df(b, start, end, vnames,case):
    file = "./"+case+"/log_%s.txt"%b
    txt = open(file, "r").readlines()
    tree = txt[start:end]

    lnode =[]
    lchar = []
    lcutp = []
    lcutq = []
    lseq = []
    for i in tree:
        if float(i.split()[2])==0:
            char = 'end'
            cutp = 'end'
            cutq = 'end'
            seqi = 'end'
        else:
            char = vnames[int(i.split()[1])]
            cutp = i.split()[2]
            cutq = int(i.split()[3])+1
            seqi = i.split()[4]
        node = i.split()[0]

        lnode.append(node)
        lchar.append(char)
        lcutp.append(cutp)
        lcutq.append(cutq)
        lseq.append(seqi)

    df = pd.DataFrame([lnode, lseq, lchar, lcutp, lcutq]).T
    df.columns = ['node','seq','char','cutp','cutq']
    df['b'] = b

    return (df, df[df['seq']!='end'][['seq','char','b']])

def get_all_bchars(B,case):
    all_bchars = []
    for b in range(1,B):
        bchars = list(read_splitting_chars(b,case))
        all_bchars = all_bchars+bchars
    return all_bchars
    
### MAIN ###
def main(b,case):
    (start,end,vnames) = find_start_end_chars(b,case)
    (df, dfs) = get_df(b, start, end, vnames, case)
    return df 

def read_splitting_chars(b,case):
    C = pd.read_csv("./"+case+"/output_%s/splitting_chars.csv"%b, index_col=0)
    return set(C['splitting_chars'])


# split variable importance

df = pd.DataFrame()

for b in range(1,B):
    # print(b)
    dfs = main(b,case)
    df=df.append(dfs)

all_bchars = get_all_bchars(B,case)
c_all=Counter(all_bchars)

# first 1 split, top 5 char

x=df[ (df['seq']=='0') ]
x=x.drop_duplicates(['b','char'])
x=x['char']
c=Counter(x)
print(c.most_common(5))

freq = pd.Series(c).sort_index()/pd.Series(c_all).sort_index()
tmp = freq.sort_values(ascending=False)
tmp.index = [i[5:] for i in tmp.index]
tmp.to_csv("top1split_"+case+".csv")
print(tmp)

# first 2 split, top 5 char¶

x=df[ (df['seq']=='0') | (df['seq']=='1')]
x=x.drop_duplicates(['b','char'])
x=x['char']
c=Counter(x)
c.most_common(5)

freq = pd.Series(c).sort_index()/pd.Series(c_all).sort_index()
tmp = freq.sort_values(ascending=False)
tmp.index = [i[5:] for i in tmp.index]
tmp.to_csv("top2split_"+case+".csv")
print(tmp)

# first 3 split, top 5 char¶

x=df[ (df['seq']=='0') | (df['seq']=='1') | (df['seq']=='2') ]
x=x.drop_duplicates(['b','char'])
x=x['char']
c=Counter(x)
c.most_common(5)

freq = pd.Series(c).sort_index()/pd.Series(c_all).sort_index()
tmp = freq.sort_values(ascending=False)
tmp.index = [i[5:] for i in tmp.index]
tmp.to_csv("top3split_"+case+".csv")
print(tmp)

