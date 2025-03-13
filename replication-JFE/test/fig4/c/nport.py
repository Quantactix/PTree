import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


case = 'yr_2001_2020__1981_2000_num_iter_9_boost_20' 
folder = 'P-Tree-c'

m = pd.read_csv("../../"+folder+"/"+case+"_months_train.csv")#, index_col=0)
p = pd.read_csv("../../"+folder+"/"+case+"_leaf_index1.csv")#, index_col=0)

df = pd.DataFrame()
df['month'] = m['months_train']
df['port'] = p['insPred1.leaf_index']
df['cons'] = 1

grp = pd.DataFrame(df.groupby(['month','port'], as_index=True).count())
grp['p'] = [i[1] for i in grp.index]
grp['m'] = [i[0] for i in grp.index]
grp.index = range(len(grp.index))

tb = grp.pivot_table(index='m', columns='p', values='cons')
print(tb.median())
tb.median().to_csv(case+"_median_port1.csv")
tb.mean().to_csv(case+"_mean_port1.csv")

