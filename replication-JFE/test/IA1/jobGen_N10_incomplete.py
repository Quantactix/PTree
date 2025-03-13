# This script generates all jobs needed
import os
import numpy as np

insert_line = 1

B = 10
trial = "PTree-2024_N10_incomplete"
case = "Oct"

# kappa = 1 #, 2, 1, 0.5

for kappa in [1, 2, 0.5]:
    output_folder = os.path.join(trial,'kappa_%s'%kappa)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(1,B+1):
        output_folder = os.path.join('./%s/kappa_%s/seq_%s'%(trial,kappa,i))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # write main file, according to template
        line_count = 1
        f0 = open('./generate_code_msr/example/B_main_N10_incomplete.R','r')
        f = open('./%s/kappa_%s/seq_%s/main.R'%(trial,kappa,i),'w')
        while line_count < insert_line:
            line_count = line_count + 1
            f.write(f0.readline())

        # write seq, kappa, case
        f.write("seq <- "+str(i)+" \n")
        f.write("kappa <- "+str(kappa)+" \n")
        f.write("case <- '"+case+"' \n")

        f.write(f0.read())
        f0.close()
        f.close()

        f1 = open('./%s/kappa_%s/sh%s.sh'%(trial,kappa,i),'w')
        f1.write("cd seq_%s \n"%(i))
        f1.write("Rscript main.R > log.txt 2>&1 \n")
        f1.write("cd ..")
        f1.close()

    # GNU parallel
    # sh 
    f2 = open('./%s/kappa_%s/submit_1.sh'%(trial,kappa),'w')
    f2.write("seq %s %s | parallel sh sh{}.sh \n"%(1,5))
    f2.close()


    f2 = open('./%s/kappa_%s/submit_2.sh'%(trial,kappa),'w')
    f2.write("seq %s %s | parallel sh sh{}.sh \n"%(6,10))
    f2.close()

    f2 = open('./%s/kappa_%s/submit.sh'%(trial,kappa),'w')
    f2.write("sh submit_1.sh \n")
    f2.write("sh submit_2.sh \n")
    f2.close()
