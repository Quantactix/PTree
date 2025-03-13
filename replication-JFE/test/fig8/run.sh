
python jobGen.py
sh submit.sh
Rscript make_big_matrix_for_all_seed.R
Rscript main_complex_sdf_regression_rolling_NSim_DT.R
Rscript plot_20240817_loop.R
python plot_20240817.py