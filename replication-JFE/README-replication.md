
# Overview

Replication package for the paper "Growing the Efficient Frontier on Panel Trees" accepted to the Journal of Financial Economics.

- Run in command line: `sh ProduceExhibits.sh` and get all results.
- The authors run the code on an Intel Xeon Gold 6230 CPU. It takes 20 minutes to fit a single Panel Tree with 61 characteristics and 2.2 million observations.
- Subfolder `P-Tree-a` takes about 7 hours, subfolder `P-Tree-b` takes about 3 hours, and subfolder `P-Tree-c`takes 3 about hours to run.
- Random P-Forest requires 1,000 P-Trees, so it takes long time to run. Subfolder `tab6` and `fig8` should take less than 24 hours, once neccessary parallel computing be applies. The same situations applies to subfoler `IA1` and `IA3`. 
- Other subfolders finish runing very quick on a PC.
- The replicator should expect the code to run for about 24 hours for each task. The authors highly recommend to apply parallel replication for different tasks in subfolders. The time consumption depends on the computing facilities.  

# Data Availability and Provenance Statements

- All input data should be stored in subfolde: `./replication-JFE/data/`, inlcuding sorted portfolios and Fama-French factors downloaded from Ken French's website: http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html, Q5 factors download from HXZ's website: http://global-q.org/factors.html, market-level time-series variables collected from FRED and Amit Goyal' website: https://sites.google.com/view/agoyal145/?redirpath=/, and IPCA factors and RPPCA factor calculated with our stock-return-characteristics data.
- The stock-return-characteristics data are from WRDS (CRSP, COMPUSTAT, and IBES). The replicator should be subscribed to these dataset. Link to raw data: https://www.dropbox.com/scl/fo/aw97m016khgk5ipyvaf5h/ADPOosy0ozN4PmfQztSSPKY?rlkey=107mrwxjb8g0iwbnvtxz0yjlw&st=w1pqxpo9&dl=0.
    Alternatively, the replicator can calculate this data via the code shared by the link: https://github.com/Feng-CityUHK/EquityCharacteristics

# Statement about Rights

- I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript.
- I DO NOT certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package.

# Summary of Availability

- Some data cannot be made publicly available.
- The only exceptoin is that: replicators should be subscribed to WRDS (CRSP, COMPUSTAT, and IBES) for full access to the stock-return-characteristics data. Alternatively, the replicator can calculate this data via the code shared by the link: https://github.com/Feng-CityUHK/EquityCharacteristics

# Details on each Data Source

| Data.Name                        | Location                                          | Provider                                          | Avaiblable             |
| -------------------------------- | ------------------------------------------------- | ------------------------------------------------- | ---------------------- |
| Stock-return-characteristic data | ./replication-JFE/data/data_202312.feather        | The authors and WRDS                              | Need WRDS Subscription |
| Portfolios returns               | ./replication-JFE/data/download_portfolios        | Ken French' website                               | Yes                    |
| IPCA factors                     | ./replication-JFE/data/IPCA                       | The authors and code from Seth Pruitt's website   | Yes                    |
| Q5 factors                       | ./replication-JFE/data/Q5                         | HXZ's website                                     | Yes                    |
| RPPCA factors                    | ./replication-JFE/data/RPPCA                      | The authors and code from Markus Pelger's website | Yes                    |
| Fama French factors              | ./replication-JFE/data/FactorsMonthly_202312.csv  | Ken French' website                               | Yes                    |
| Macroeconomic variables          | ./replication-JFE/data/xt_1972_2021_10y_stand.csv | Amit Goyal's website and FRED                     | Yes                    |

# Description of programs/code

- Please read `ProduceExhibits.sh`. Each task/experiment is stored in a subfolder, for instance, `P-Tree-a`.
- In each subfolder, there is a `run.sh` file to replicate all results for this task/experiment. The output csv or pdf files are in saved this subfolder. Some subfolders also include an Excel file for demonstration.



The provided code reproduces:

- All numbers provided in text in the paper
- All tables and figures in the paper

# Reference

@article{cong2025ptree,

    title={Growing the efficient frontier on panel trees},

    ​author={Cong, Lin William and Feng, Guanhao and He, Jingyu and He, Xin},

    ​journal={Journal of Financial Economics, forthcoming},

    ​year={2025},

    Volume={167}

​}
