# This script generates all jobs needed
import os
import numpy as np
np.random.seed(20230928)

insert_line = 1

# generate random choices from a list

def gen_chars():
    # without replacement
    candidate = list(range(1,62))
    rd = np.random.choice(candidate, size=20, replace=False)
    rd = [str(i) for i in rd]
    return rd

def gen_dates(): 
    # with replacement
    n_day = 480 
    candidate = list(range(1,n_day+1))
    rd = np.random.choice(candidate, size=n_day, replace=True)
    rd = [str(i) for i in rd]
    return rd

B = 1000+1
batch = 20

for i in range(1,B):
    output_folder = os.path.join('output_%s'%i)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # write main file, according to template
    line_count = 1
    f0 = open('rf_template-invest.R','r')
    f = open('main_%s.R'%i,'w')
    while line_count < insert_line:
        line_count = line_count + 1
        f.write(f0.readline())

    # generate random chars
    rd = gen_chars()
    f.write("random_chars <- c("+",".join(rd)+") \n")
    rdd = gen_dates()
    f.write("random_dates <- c("+",".join(rdd)+") \n")
    f.write("case <- '%s' \n"%i)
    f.write(f0.read())
    f0.close()
    f.close()

    f1 = open('sh%s.sh'%i,'w')
    f1.write("Rscript main_%s.R > log_%s.txt 2>&1"%(i,i))
    f1.close()

# GNU parallel
# sh 
f2 = open('submit_1.sh','w')
i = 1
while i < (B-1)/2:
    j = min(i+batch-1,B-1)
    f2.write("seq %s %s | parallel sh sh{}.sh \n"%(i,j))
    i = i+batch

f2.close()


f2 = open('submit_2.sh','w')
i = B-1
while i > B/2:
    j = min(i-batch+1,B)
    # print(j,i)
    f2.write("seq %s %s | parallel sh sh{}.sh \n"%(j,i))
    i = i-batch

f2.close()

f2 = open('submit.sh','w')
f2.write("seq 1 2 | parallel sh submit_{}.sh \n")
f2.close()