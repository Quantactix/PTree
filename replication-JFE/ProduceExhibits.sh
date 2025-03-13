############################################################################
# This sh file is to reproduce empirical results for paper "Growing the Efficient Frontier on Panel Trees" accepted to the Journal of Financial Economics.
# More updates on Panel Tree package are hosted on Github repo: https://github.com/Quantactix/PTree.
############################################################################

############################################################################
# Caveate.
# Please make sure that the R package "PTree" be installed, before running this script.
############################################################################

cd ./test/

############################################################################
# P-Trees trained for 1981 to 2020. 
############################################################################

# Train boosted P-Trees.
cd ./P-Tree-a/
sh run.sh
cd ..

############################################################################
# P-Trees trained for 1981 to 2000, tested by 2001 to 2020. 
############################################################################

# Train boosted P-Trees.
cd ./P-Tree-b/
sh run.sh
cd ..

############################################################################
# P-Trees trained for 2001 to 2020, tested by 1981 to 2000. 
############################################################################

# Train boosted P-Trees.
cd ./P-Tree-c/
sh run.sh
cd ..

############################################################################
# Plot P-Trees Diagram.
# Figures 4, A.1 (a), A.1 (b).
############################################################################

# Figure 4: Panel Tree from 1981 to 2020
cd ./fig4/a/
sh run.sh

# Figure A.1: Panel Tree Diagram for Subsamples
# (a) 20-year Sample (1981-2000)
cd ../b/
sh run.sh

# Figure A.1: Panel Tree Diagram for Subsamples
# (b) 20-year Sample (2001-2020)
cd ../c/
sh run.sh
cd ..
cd ..

############################################################################
# Table 1: Evaluation for Leaf Basis Portfolios.
############################################################################

cd ./tab1/
sh run.sh
cd ..

############################################################################
# Figure 6: Diversified P-Tree Test Assets
# Figure A1, A2, A3, B1, B2, B3
############################################################################

cd ./fig6/
sh run.sh
cd ..

############################################################################
# Figure 7: Characterizing the Efficient Frontier with P-Trees
############################################################################

cd ./fig7/
sh run.sh
cd ..

############################################################################
# Table 2: Testing the Boosted P-Tree Growth
# Table A.3: Subsample Analysis for Testing the Boosted P-Tree Growth
############################################################################

cd ./tab2/
sh run.sh
cd ..

############################################################################
# Table 3: Comparing Test Assets
############################################################################

cd ./tab3/
sh run.sh
cd ..

############################################################################
# Table 4: Asset Pricing Performance: Cross-Sectional R2
############################################################################

cd ./tab4/
sh run.sh
cd ..

############################################################################
# Table 5: Factor Investing by Boosted P-Trees
############################################################################

cd ./tab5/
sh run.sh
cd ..

############################################################################
# Table 6: Characteristic Importance by Selection Probability
# This step involves training a Random P-Forest with 1,000 P-Trees. 
# The replication code adopts GNU parallel for parallel computing of P-Trees.
# The computation time is quite long, depending on your devices.
############################################################################

cd ./tab6/
sh run.sh
cd ..

############################################################################
# Figure 8: OOS Performance of Random P-Forest SDF
############################################################################

cd ./fig8/
sh run.sh
cd ..

############################################################################
# Figure 10: Evaluating a Characteristic with P-Tree
############################################################################

cd ./fig10/
sh run.sh
cd ..

############################################################################
# Table 7: P-Tree Performance Under Regime Switches
############################################################################

cd ./tab7/
sh run.sh 
cd ..

# Cannot Replicate. We can present the most updated results.

############################################################################
# Internet Appendices I. Simulation.
# Table I.1 to I.4.
# The computation time is quite long, depending on your devices.
############################################################################

cd ./IA1/
sh run.sh
cd ..

############################################################################
# Internet Appendices II. Benchmark-Adjusted P-Trees.
# Figure I.1.
# Table I.5 to I.7.
############################################################################

cd ./IA2/
sh run.sh
cd ..

############################################################################
# Internet Appendices III. Over-parameterized SDF via Random Split.
# Figure I.2.
# Table I.8 to I.9.
# The computation time is quite long, depending on your devices.
############################################################################

cd ./IA3/
sh run.sh
cd ..

############################################################################
# Contact 
############################################################################
# Xin He 
# Email: xin.he@ustc.edu.cn 
# Homepage: www.xinhesean.com 
############################################################################
