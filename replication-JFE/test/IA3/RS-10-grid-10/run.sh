
# generate random seed and multiple simulations.
python jobGen.py 

# submit code for random split P-Trees.
sh submit.sh

# combine basis portfolios.
Rscript make_big_matrix_for_a_seed.R 
Rscript make_big_matrix_for_all_seed.R 

# run SDF ridge regression.
Rscript main_complex_sdf_regression_rolling_NSim_DT.R 

# plot
Rscript plot_20240516_loop.R 
