# 20230708 add a new sh to extract the basis portfolios and save in csv

# This script generates all jobs needed
import os
import numpy as np
np.random.seed(20230705)

xt_list = ['x_svar', 'x_infl','x_dy', 'x_lev', 'x_ep', 
            'x_ni','x_tbl', 'x_dfy', 'x_tms', 'x_ill']


f2 = open("submit_basis_portfolio_allportfolio.sh",'w')

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
    f2.write("sh submit_basis_portfolio.sh \n")
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

    insert_line_one = 5
    line_count = 0
    f0 = open("./template/main-m2-basis-portfolio.R",'r')
    f = open(os.path.join(fld,'main-m2-basis-portfolio-case.R'),'w')
    while line_count < insert_line_one:
        line_count = line_count + 1
        f.write(f0.readline())

    f.write(f0.read())
    f.write("write.csv(data.frame(port), '_basis_portfolio_btm_.csv')")
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

    insert_line_one = 5
    line_count = 0
    f0 = open("./template/main-m2-basis-portfolio.R",'r')
    f = open(os.path.join(fld,'main-m2-basis-portfolio-case.R'),'w')
    while line_count < insert_line_one:
        line_count = line_count + 1
        f.write(f0.readline())

    f.write(f0.read())
    f.write("write.csv(data.frame(port), '_basis_portfolio_top_.csv')")
    f0.close()
    f.close()

    ########################
    # write sh file for each xt #
    ########################   

    ##  btm
    f1 = open(os.path.join(xt,"sh-basis-portfolio-1.sh"),'w')
    f1.write("cd btm \n")
    f1.write("Rscript main-m2-basis-portfolio-case.R > log-basis-portfolio.txt 2>&1")
    f1.close()

    # top
    f1 = open(os.path.join(xt,"sh-basis-portfolio-2.sh"),'w')
    f1.write("cd top \n")
    f1.write("Rscript main-m2-basis-portfolio-case.R > log-basis-portfolio.txt 2>&1")
    f1.close()

    f1 = open(os.path.join(xt,"submit_basis_portfolio.sh"),'w')
    f1.write("sh sh-basis-portfolio-1.sh \n")
    f1.write("sh sh-basis-portfolio-2.sh \n")
    f1.close()

f2.close()