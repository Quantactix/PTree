
set.seed(s)

t = proc.time()

###### parameters #####

case='RANDOM_yr_1981_2000__2001_2020_num_iter_9_boost_10' 

start = '1981-01-01' 
split = '2000-12-31' 
end = '2020-12-31' 

min_leaf_size = 20 
max_depth = 10
max_depth_boosting = 10

num_iter = 99
num_iterB = 99

num_cutpoints = 4
equal_weight = FALSE

no_H1 = TRUE # TRUE, FALSE
no_H = FALSE # TRUE, FALSE

abs_normalize = TRUE
weighted_loss = FALSE

early_stop = FALSE
stop_threshold = 1
lambda_ridge = 0

lambda_mean = 0
lambda_cov = 1e-4

lambda_mean_factor = 0
lambda_cov_factor = 1e-5

## parameters for extension, but redundant for P-Tree paper. 
eta=1

range_constraint = FALSE
upper_limit = Inf
lower_limit = 0
gamma_optimize = 2.0
sum_constraint = 1.0

a1=0.0
a2=0.0
list_K = matrix(rep(0,3), nrow = 3, ncol = 1)

random_split = TRUE

###### library #####

library(PTree)
library(rpart)
library(ranger)

##### load data #####

library(arrow)
my_tibble = arrow::read_feather("../../../../data/data_202312.feather")
data <- as.data.frame(my_tibble)

# re-order the data
tmp = data[,c('gvkey', 'permno', 'sic', 'ret', 'exchcd', 'shrcd', 'date', 'ffi49',
              'lag_me', 
              
              'rank_me', 'rank_bm', 'rank_agr', 'rank_op', 'rank_mom12m',
              
              'rank_re', 
              'rank_mom1m', 'rank_beta', 'rank_std_dolvol',
              'rank_std_turn',  'rank_depr', 'rank_ni',
              'rank_roe', 'rank_hire', 'rank_pm', 'rank_turn', 'rank_acc', 'rank_ep',
              'rank_ill', 'rank_ato', 'rank_cashdebt', 'rank_rvar_capm',
              'rank_chpm', 'rank_adm', 'rank_baspread', 'rank_alm', 
              'rank_sp', 'rank_dy', 'rank_rdm', 'rank_me_ia', 'rank_nincr',
              'rank_bm_ia', 'rank_maxret', 'rank_zerotrade', 'rank_noa', 'rank_cfp',
              'rank_mom36m', 'rank_gma',  'rank_lgr', 'rank_rna',
              'rank_mom60m', 'rank_roa', 'rank_herf', 'rank_cash', 'rank_rd_sale',
              'rank_svar', 'rank_abr', 'rank_sgr', 'rank_seas1a', 'rank_rsup',
              'rank_cinvest', 'rank_grltnoa', 'rank_sue', 'rank_mom6m', 'rank_chcsho',
              'rank_lev', 'rank_rvar_ff3', 'rank_dolvol', 'rank_pscore',
              'rank_pctacc', 'rank_chtx', 
              
              'log_me', 
              
              'MKTRF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 
              
              'x_tbl', 'x_dfy', 'x_tms', 'x_svar', 'x_ill', 'x_infl',
              'x_dy', 'x_lev', 'x_ep', 'x_ni',
               
              'xret')] 

data = tmp
rm(tmp)

# chars 

all_chars <- names(data)[c(10:70)]
top5chars <- c(1:5)
instruments = all_chars[top5chars]
splitting_chars <- all_chars

first_split_var = c(1:61)-1
second_split_var = c(1:61)-1

##### train-test split #####

data[,c('date')] <- as.Date(data[,c('date')], format='%Y-%m-%d')
data1 <- data[(data[,c('date')]>=start) & (data[,c('date')]<=split), ]
data2 <- data[(data[,c('date')]>split) & (data[,c('date')]<=end), ]

###### train data for all boosting steps #####
X_train = data1[,splitting_chars]
R_train = data1[,c("xret")]
months_train = as.numeric(as.factor(data1[,c("date")]))
months_train = months_train - 1 # start from 0
stocks_train = as.numeric(as.factor(data1[,c("permno")])) - 1
Z_train = data1[, instruments]
Z_train = cbind(1, Z_train)
# Z_train = 0
portfolio_weight_train = data1[,c("lag_me")]
loss_weight_train = data1[,c("lag_me")]
num_months = length(unique(months_train))
num_stocks = length(unique(stocks_train))


###### BENCHMARK FACTORS ##### 

# mkt
ff = read.csv('../../../../data/FactorsMonthly_202312.csv', row.names = 1)
names(ff) <- c('mktrf','smb','hml','rmw','cma','rf','mom')
ff <- ff/100
row.names(ff) <- as.Date(paste0(row.names(ff), '01'), format='%Y%m%d')

ff_train <- ff[row.names(ff)>=start,]
ff_train <- ff_train[row.names(ff_train)<=split,]
avg_ff_train <- colMeans(ff_train)

ff_test <- ff[row.names(ff)>split,]
ff_test <- ff_test[row.names(ff_test)<=end,]
avg_ff_test <- colMeans(ff_test)


###### test data #####

X_test = data2[,splitting_chars]
R_test = data2[,c("xret")]
months_test = as.numeric(as.factor(data2[,c("date")]))
months_test = months_test - 1 # start from 0
stocks_test = as.numeric(as.factor(data2[,c("permno")])) - 1
Z_test = data2[,instruments]
Z_test = cbind(1, Z_test)
H_test = data2[,c("MKTRF")]
H_test = H_test * Z_test
portfolio_weight_test = data2[,c("lag_me")]
loss_weight_test = data2[,c("lag_me")]

###### train #####
# the first `H`` is the `mkt``, but we use `no_H1 = TRUE`, so the algorithm ignores the `H`.
Y_train1 = data1[,c("xret")]
H_train1 = ff_train$mktrf

B = 10

for (b in 1:B){
  print(b)
  fit1 = PTree(R_train, Y_train1, X_train, Z_train, H_train1, portfolio_weight_train, 
  loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, num_stocks, 
  num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
  no_H1, 
  abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

  pred1 = predict(fit1, X_test, R_test, months_test, portfolio_weight_test)

  save(fit1,  file = paste0("../trees/seed_",s,"/random_train_",b,".RData"))
  save(pred1, file = paste0("../trees/seed_",s,"/random_test_",b,".RData"))

}