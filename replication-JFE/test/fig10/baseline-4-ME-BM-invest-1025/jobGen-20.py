# This script generates all jobs needed
import os
# from datetime import *
import numpy as np
np.random.seed(20231025)

insert_line = 1

for i in range(3,62):
    print(i)
    
    output_folder = os.path.join('char_%s'%i)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # write main file, according to template
    line_count = 1

    j = 1
    f0 = open('b4.R','r')
    f = open(os.path.join('char_%s'%i,'main_%s.R'%j),'w')
    while line_count < insert_line:
        line_count = line_count + 1
        f.write(f0.readline())

    # insert char id
    f.write("i <- %s \n"%i)

    f.write(f0.read())
    f0.close()
    f.close()

    f1 = open('sh_char_%s.sh'%(i),'w')
    f1.write("Rscript ./char_%s/main_%s.R > ./char_%s/log_%s.txt 2>&1"%(i,j,i,j))
    f1.close()

# GNU parallel
# sh 

f2 = open('submit.sh','w')

print(i)
f2.write("seq 3  20 | parallel sh sh_char_{}.sh \n")
f2.write("seq 21 40 | parallel sh sh_char_{}.sh \n")
f2.write("seq 41 50 | parallel sh sh_char_{}.sh \n")
f2.write("seq 51 61 | parallel sh sh_char_{}.sh \n")

f2.close()