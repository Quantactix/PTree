
# generate random seed and multiple simulations.
python jobGen.py 

# submit code for random split P-Trees.
sh submit.sh

# combine basis portfolios.
Rscript make_big_matrix_for_a_seed.R > log.make_big_matrix_for_a_seed.R.txt 2>&1
Rscript make_big_matrix_for_all_seed.R > log.make_big_matrix_for_all_seed.R.txt 2>&1 

# run SDF ridge regression.
Rscript main_complex_sdf_regression_rolling_NSim_DT.R > log.main_complex_sdf_regression_rolling_NSim_DT.R.txt 2>&1 

# plot
Rscript plot_20240516_loop.R > log.plot_20240516_loop.R.txt 2>&1 
python plot_20240516.py > log.plot_20240516.py.txt 2>&1 
