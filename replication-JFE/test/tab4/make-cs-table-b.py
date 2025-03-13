
import pandas as pd
import numpy as np

folder  = 'b'

################## without the first row. ##################

da = pd.read_csv("./performance/Portfolio_R2_P-Tree-"+folder+".csv", index_col=0)

# ins
out = pd.pivot_table(da, index='port', columns='model_list', values='inscs')
out = out.loc[['pt5','pt10', 'pt20', 'unisort', 'bisort', 'mebm25', 'ind49']]
out.index = ['PTree1-5','PTree1-10','PTree1-20','Uni-Sort','Bi-Sort','ME/BM','Ind49']
out = out[["PT1","PT5","PT10","PT20",'FF5','Q5','RP5','IP5']]
out.columns = ['P-Tree1F','P-Tree5F','P-Tree10F','P-Tree20F','FF5','Q5','RP5','IP5']
cs_ins = out

################## with the first row. ##################

da = pd.read_csv("./performance/Portfolio_R2_P-Tree-"+folder+"-2.csv", index_col=0)

# ins
out = pd.pivot_table(da, index='port', columns='model_list', values='inscs')
out = out.loc[['pt1']]
out['PT10'] = 'NaN'
out['PT20'] = 'NaN'
out = out[["PT1","PT5","PT10","PT20",'FF5','Q5','RP5','IP5']]
out.columns = ['P-Tree1F','P-Tree5F','P-Tree10F','P-Tree20F','FF5','Q5','RP5','IP5']
pt1 = out

# output
bigout = pt1.append(cs_ins)
bigout.to_csv("table/tab-cs-"+folder+".csv") 
