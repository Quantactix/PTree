import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


case = 'yr_2001_2020__1981_2000_num_iter_9_boost_20' 
folder = 'P-Tree-c'

# read the nport file
nport = pd.read_csv(case+"_median_port1.csv",index_col=0)['0']
print(nport)

file = "../../"+folder+"/log.txt"
f = open(file, "r")

num_lines = sum(1 for line in f)
print(num_lines)
f.close()

### find the start and end of tree structure

f = open(file, "r")

start=0
end=0
i = 0

flag1=False
flag2=False

for x in f:
    if x == 'fitted tree \n':
        print('#comments# start the nodes')
        flag1=True

    if x[:2] == '1 ' and flag1==True:
        print('#comments# node 1')
        start = i
        print(i)
        flag2 = True

    if x == '\n' and flag2==True:
        print('#comments# end the nodes')
        end = i
        print(i)
        break
    print(x)
    i=i+1

print('start of nodes: row %s'%start)
print('end of nodes: row %s'%end)

### read the tree node, char, cut point

# start: variable names

vnames = ['rank_me', 'rank_bm', 'rank_agr', 'rank_op', 'rank_mom12m',
        
        # 15
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
        'rank_rvar_mean', 'rank_abr', 'rank_sgr', 'rank_seas1a', 'rank_rsup',
        'rank_cinvest', 'rank_grltnoa', 'rank_sue', 'rank_mom6m', 'rank_chcsho',
        'rank_lev', 'rank_rvar_ff3', 'rank_dolvol', 'rank_pscore',
        'rank_pctacc', 'rank_chtx'
]

print(vnames)
print(len(vnames))

# end: variable names

txt = open(file, "r").readlines()
tree = txt[start:end]

lnode =[]
lsplt =[]
lchar = []
lcutp = []
for i in tree:
    print(i)
    if float(i.split()[2])==0:
        char = 'end'
        cutp = 'end'
        splt = 'end'
    else:
        char = vnames[int(i.split()[1])]
        cutp = i.split()[2]
        splt = i.split()[4]
        splt = str(int(splt)+1)
    node = i.split()[0]

    lnode.append(node)
    lchar.append(char)
    lcutp.append(cutp)
    lsplt.append(splt)


### write latex tree plot

# lnode = ['1', '2', '4', '8', '9', '5', '3', '6', '12', '13', '7', '14', '15']
# lchar = ['rank_me', 'rank_op', 'rank_cash', 'end', 'end', 'end', 'rank_me', 'rank_chpm', 'end', 'end', 'rank_pm', 'end', 'end']
# lcutp = ['0.2', '0.8', '0.4', 'end', 'end', 'end', '0.4', '0.4', 'end', 'end', '0.8', 'end', 'end']

lchar = [i[5:].upper() for i in lchar]

def write_row(node, splt, char, cutp, n_indent=0):
    f.write(n_indent*2*"   "+"child { node [env] {N%s  S%s \\\\ %s $\leq$ %s} \n"%(node, splt, char, cutp))

# def write_row_leaf(node, splt, char, cutp, n_indent=0):
#     f.write(n_indent*2*"   "+"child { node [env] {N%s \\\\ } \n"%(node))


# n = len(lnode)

# f = open("tree.tex", "w+")

# i=0
# f.write("\\node [env] {N%s \\\\ %s $\leq$ %s } \n"%(lnode[i], lchar[i], lcutp[i]))
# while i < n-1:
#     i = i+1
#     # print(i)
#     n_indent = int(np.floor(np.log2(int(lnode[i]))))
#     # print(n_indent)
#     write_row( lnode[i], lchar[i], lcutp[i] , n_indent)

# f.close()

# recursion

f = open("tree.tex", "w+")
f.write("\n"+"%"+"start python generated \n")

lintnode = [int(i) for i in lnode]

node = 1
print("i")
print(i)
print("lnode")
print(lnode)

i = lnode.index(str(node))
f.write("\\node [env] {N%s  S%s\\\\ %s $\leq$ %s } \n"%(lnode[i], lsplt[i], lchar[i], lcutp[i]))

def recursion(node):
    # f.write("%"+"%s \n"%node)

    l = node*2
    r = node*2+1

    if l in lintnode:
        # f.write('%'+' node %s have left child \n'%l)
        i = lnode.index(str(l))
        n_indent = int(np.floor(np.log2(int(lnode[i]))))
        write_row(lnode[i], lsplt[i], lchar[i], lcutp[i], n_indent)

        # write the left child
        recursion(l)
        f.write((1+n_indent)*2*"   "+"edge from parent node [left] {Y} \n")
        f.write(n_indent*2*"   "+"}\n")
    else:
        # f.write('%'+' node %s do not have left child \n'%l)
        tmp = 1

    if r in lintnode:
        # f.write('%'+' node %s have right child \n'%r)
        i = lnode.index(str(r))
        write_row(lnode[i], lsplt[i], lchar[i], lcutp[i], int(np.floor(np.log2(int(lnode[i])))))

        # write the right child
        recursion(r)
        f.write((1+n_indent)*2*"   "+"edge from parent node [right] {N} \n")
        f.write(n_indent*2*"   "+"}\n")
    else:
        # f.write('%'+' node %s do not have right child \n'%r)
        tmp = 1

    # f.write('%'+' node %s finish \n'%node)

recursion(node)

f.write("\n"+"%"+"end python generated \n")
f.close()

# replace end

f = open("tree.tex", "r")
f2 = open("tree2.tex", "w+")

for x in f:
    if  "Send \\\\  $\leq$ end" in x:            
        N = int(x.split()[4][2:])
        # print(x)
        # print(x.split())
        # print("Node %s is end"%(N))
        nStock = nport.loc[N]
        print("N Stock is %s"%(nStock))
        txt = x.replace("Send \\\\  $\leq$ end"," \\\\ \# %.0f "%(nStock))
    else:
        txt = x
    txt = txt.replace("_","\_")
    f2.write(txt)

f.close()
f2.close()

