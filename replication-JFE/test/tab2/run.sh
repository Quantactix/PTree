python tab2_a.py 
python tab2_b.py 
python tab2_c.py 

cd ./BS/
matlab -nodisplay -nosplash -nodesktop -r "run('BS_a.m');exit;"
matlab -nodisplay -nosplash -nodesktop -r "run('BS_b.m');exit;"
matlab -nodisplay -nosplash -nodesktop -r "run('BS_c.m');exit;"
cd ..
