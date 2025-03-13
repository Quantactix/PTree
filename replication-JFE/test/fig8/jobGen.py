# This script generates all jobs needed
import os
import numpy as np
np.random.seed(20240430)

insert_line = 1

N_Seed = 100000
batch  = 20

# generate random choices from a list

def gen_chars():
    # without replacement
    candidate = list(range(1,62))
    # print(candidate)
    rd = np.random.choice(candidate, size=10, replace=False) # select 5 chars out of 61
    rd = [str(i) for i in rd]
    return rd

for s in range(1,N_Seed+1):
    output_folder = os.path.join('trees/seed_%s'%s)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # write main file, according to template
    line_count = 1
    f0 = open('main_sample.R','r')
    f = open('./code/main_seed_%s.R'%s,'w')
    while line_count < insert_line:
        line_count = line_count + 1
        f.write(f0.readline())

    # write seed number
    f.write("s <- "+str(s)+" \n")

    # generate random chars
    rd = gen_chars()
    f.write("random_chars <- c("+",".join(rd)+") \n")

    # write the other lines
    f.write(f0.read())
    f0.close()
    f.close()

    f1 = open('./code/sh%s.sh'%s,'w')
    f1.write("Rscript main_seed_%s.R > log_%s.txt 2>&1"%(s,s))
    f1.close()

# GNU parallel
# sh 
f2 = open('submit.sh','w')
f2.write("cd code \n")
i = 1
while i < N_Seed:
    j = min(i+batch-1,N_Seed)
    f2.write("seq %s %s | parallel sh sh{}.sh \n"%(i,j))
    i = i+batch

f2.write("cd .. \n")
f2.close()

