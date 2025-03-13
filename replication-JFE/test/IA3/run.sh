
cd ./RS-10/
sh run.sh # get Figure I.2.

# other cases with the number of leaves in a P-Tree be 2, 5, 10, 20, 100.
# code is similar to "./RS-10/".
# Need to revise the parameter "num_iter" in "main_sample.R".
# get Table I.8.
# see csv files in subfoler "output" of following specifications.

cd ../RS-2/
sh run.sh

cd ../RS-5/
sh run.sh

cd ../RS-20/
sh run.sh

cd ../RS-100/
sh run.sh

# robustness for Threshold Grid.
# Code is similar to "./RS-10/".
# Need to revise the parameter "num_cutpoints" in "main_sample.R".
# get Table I.9.
# see csv files in subfoler "output" of following specifications.

cd ../RS-10-grid-3/
sh run.sh

cd ../RS-10-grid-5/
sh run.sh

cd ../RS-10-grid-10/
sh run.sh

cd ..
