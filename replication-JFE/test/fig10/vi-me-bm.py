#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
loss_base = 1.3714529580881745  # the baseline is the Sharpe ratio of Figure 9: (a) ME Baseline P-Tree.

### find the loss and improvement of a char ###
def find_numbers(case,b=1):

    # check if killed
    file = case+"/log_%s.txt"%b
    f = open(file, "r")
    i = 0
    for x in f:
        if "Killed" in x:
            return (-100, -100, -100)
    
    # find the line number of keywords
    file = case+"/log_%s.txt"%b
    f = open(file, "r")
    i = 0
    for x in f:
        if "Criteria Base End" in x:
            line_base_loss = i
        if "Criteria Char End" in x:
            line_char_loss = i
        i=i+1

    # find the key numbers
    file = case+"/log_%s.txt"%b
    f = open(file, "r")
    i = 0
    for x in f:
        if i==line_char_loss-1:
            char_loss = float(x[4:-1])
        i=i+1
    return char_loss

def evaluate_a_char(i):
    case = "./baseline-4-ME-BM-invest-1025/char_%s"%i
    char_loss = find_numbers(case,b=1)
    
    return - char_loss

### loop over chars ###
def main():
    list_sr = []
    for i in range(3,62):
        sr = evaluate_a_char(i)
        list_sr.append(sr)
    return list_sr
        
loss_char = main()
loss_all = [loss_base] + list(loss_char)
loss_all

# plot

all_chars = [
'rank_me', 'rank_bm', 'rank_agr', 'rank_op', 'rank_mom12m',

'rank_re', 
'rank_mom1m', 'rank_beta', 'rank_std_dolvol',
'rank_std_turn',  'rank_depr', 'rank_ni',
'rank_roe', 'rank_hire', 'rank_pm', 'rank_turn', 'rank_acc', 'rank_ep',
'rank_ill', 'rank_ato', 'rank_cashdebt', 'rank_rvar_capm',
'rank_chpm', 'rank_adm', 'rank_baspread', 'rank_alm', 
'rank_sp', 'rank_dy', 'rank_rdm', 'rank_me_ia', 'rank_nincr',
'rank_bm_ia', 'rank_maxret', 'rank_zerotrade', 'rank_noa', 'rank_cfp',
'rank_mom36m', 'rank_gma',  'rank_lgr', 'rank_rna',
'rank_mom60m', 'rank_roa', 'rank_herf', 'rank_cash', 'rank_rd_sale',
'rank_svar', 'rank_abr', 'rank_sgr', 'rank_seas1a', 'rank_rsup',
'rank_cinvest', 'rank_grltnoa', 'rank_sue', 'rank_mom6m', 'rank_chcsho',
'rank_lev', 'rank_rvar_ff3', 'rank_dolvol', 'rank_pscore',
'rank_pctacc', 'rank_chtx'
]
all_chars = [i[5:] for i in all_chars]


s = pd.DataFrame(index=all_chars[2:])
s['bar']=loss_char
s=s.sort_values('bar')
s['t'] = 1

this_color = np.where(s['bar']>=loss_base,'black','lightgrey')
idx = [i.upper() for i in s.index]
plt.figure(figsize=(10,5), tight_layout=True, frameon=False)
plt.margins(.01)
plt.bar(height = s['bar'], x=idx, color=this_color)
plt.xticks(rotation=-90)
plt.yticks(rotation=-90)
plt.ylim((-0.1,5.0))
plt.hlines(y=loss_base,xmin=-1, xmax=59,color='red')
plt.hlines(y=0, xmin=-1, xmax=59, color='black')
plt.savefig("bar_1ME-2BM-3Char-INVEST-raw-nobench.pdf", bbox_inches = 'tight')
plt.show()
plt.close()
