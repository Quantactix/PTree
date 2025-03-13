import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
# import pyarrow.feather as pf

case = 'yr_2001_2020__1981_2000_num_iter_9_boost_20' 
folder = 'P-Tree-c'

leaf_idx = pd.read_csv("../../"+folder+"/"+case+"_leaf_index1.csv")#,index_col=0)
leaf_idx = leaf_idx['insPred1.leaf_index']
print(leaf_idx)
print(list(set(leaf_idx)))

