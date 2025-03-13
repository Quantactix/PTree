# This script generates all jobs needed
import os
# from datetime import *
import numpy as np
np.random.seed(20230705)

xt_list = ['x_svar', 'x_infl','x_dy', 'x_lev', 'x_ep', 
            'x_ni','x_tbl', 'x_dfy', 'x_tms', 'x_ill']

f2 = open("submit_all.sh",'w')

for xt in xt_list:

    ########################
    # macro variable folder #
    ########################
    fld = os.path.join(xt)
    if not os.path.exists(fld):
        os.makedirs(fld)

    ########################
    # write sh file for loop on xt #
    ########################   
    
    f2.write("echo ' "+xt+" ' \n")
    f2.write("cd "+fld+" \n")
    f2.write("sh submit.sh \n")
    f2.write("cd .. \n")
    f2.write("\n")

    ########################
    # bottom folder #
    ########################
    fld = os.path.join(xt,"btm")
    if not os.path.exists(fld):
        os.makedirs(fld)

    # write main file, according to template
    ## bottom

    insert_line_one = 131
    line_count = 0
    f0 = open("./template/main-m2.R",'r')
    f = open(os.path.join(fld,'main-m2-case.R'),'w')
    while line_count < insert_line_one:
        line_count = line_count + 1
        f.write(f0.readline())

    #### split data sample by macro condition
    f.write("data <- data[data[,c('"+xt+"')]<=0.5 ,] \n")

    insert_line_two = 138
    while line_count < insert_line_two:
        line_count = line_count + 1
        f.write(f0.readline())

    #### split time sample by macro condition
    f.write("train_ts_idx = xt_train$"+xt+"<=0.5 \n")

    f.write(f0.read())
    f0.close()
    f.close()

    ########################
    # top folder #
    ########################
    fld = os.path.join(xt,"top")
    if not os.path.exists(fld):
        os.makedirs(fld)

    # write main file, according to template
    ## bottom

    insert_line_one = 131
    line_count = 0
    f0 = open("./template/main-m2.R",'r')
    f = open(os.path.join(fld,'main-m2-case.R'),'w')
    while line_count < insert_line_one:
        line_count = line_count + 1
        f.write(f0.readline())

    #### split data sample by macro condition
    f.write("data <- data[data[,c('"+xt+"')]>=0.5 ,] \n")

    insert_line_two = 138
    while line_count < insert_line_two:
        line_count = line_count + 1
        f.write(f0.readline())

    #### split time sample by macro condition
    f.write("train_ts_idx = xt_train$"+xt+">=0.5 \n")

    f.write(f0.read())
    f0.close()
    f.close()

    ########################
    # write sh file for each xt #
    ########################   

    ##  btm
    f1 = open(os.path.join(xt,"sh-1.sh"),'w')
    f1.write("cd btm \n")
    f1.write("Rscript main-m2-case.R > log.txt 2>&1")
    f1.close()

    # top
    f1 = open(os.path.join(xt,"sh-2.sh"),'w')
    f1.write("cd top \n")
    f1.write("Rscript main-m2-case.R > log.txt 2>&1")
    f1.close()

    f1 = open(os.path.join(xt,"submit.sh"),'w')
    f1.write("seq 1 2 | parallel sh sh-{}.sh \n")
    f1.close()

f2.close()