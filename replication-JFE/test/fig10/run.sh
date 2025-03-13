
cd baseline-4-ME-ME-invest-1025
python jobGen-20.py
sh submit.sh
cd ..
python vi-me-me.py 

cd baseline-4-ME-BM-invest-1025
python jobGen-20.py
sh submit.sh
cd ..
python vi-me-bm.py 