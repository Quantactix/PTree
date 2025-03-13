
# simulate data.
cd ./code-simudate-data/
sh simu.sh

# complete signal.
python jobGen_N10.py
cd ./PTree-2024_N10/
cd ../kappa_0.5
sh submit.sh
cd ./kappa_1/
sh submit.sh
cd ../kappa_2
sh submit.sh
cd ..
python PTree-summary-kappa-0.5.py
python PTree-summary-kappa-1.py
python PTree-summary-kappa-2.py
python selected-chars.py
cd ..

# incomplete signal.
python jobGen_N10_incomplete.py
cd ./PTree-2024_N10_incomplete/
cd ../kappa_0.5
sh submit.sh
cd ./kappa_1/
sh submit.sh
cd ../kappa_2
sh submit.sh
cd ..
python PTree-summary-kappa-0.5.py
python PTree-summary-kappa-1.py
python PTree-summary-kappa-2.py
python selected-chars.py
cd ..

# true signal.
python jobGen_N10_TruePred.py
cd ./PTree-2024_N10_TruePred/
cd ../kappa_0.5
sh submit.sh
cd ./kappa_1/
sh submit.sh
cd ../kappa_2
sh submit.sh
cd ..
python PTree-summary-kappa-0.5.py
python PTree-summary-kappa-1.py
python PTree-summary-kappa-2.py
python selected-chars.py
cd ..

# an excel file is provided for tables.