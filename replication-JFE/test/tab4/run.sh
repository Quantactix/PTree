
cp -r ../tab3/tmp/ ./tmp/

Rscript main_a.R   
Rscript main_a-2.R 

Rscript main_b.R   
Rscript main_b-2.R 

Rscript main_c.R   
Rscript main_c-2.R 

python make-cs-table-a.py
python make-cs-table-b.py
python make-cs-table-c.py
