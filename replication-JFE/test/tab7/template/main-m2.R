
case='G-m2' 

start = '1981-01-01' 
split = '2020-12-31' 

min_leaf_size = 20 
max_depth = 10
max_depth_boosting = 10

num_iter = 9
num_iterB = 9

num_cutpoints = 4
equal_weight = FALSE

no_H1 = TRUE
no_H = FALSE

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

random_split = FALSE

###### library #####

library(PTree)
library(rpart)
library(ranger)

R2.calc = function(y, ypred){
  return(100*(1 - sum((y - ypred)^2) / sum(y^2)))
}

PeR2.calc = function(stocks_index, y, haty){
  df <- data.frame(cbind(stocks_index, y, haty))
  df$e=df$y-df$haty
  numer <- aggregate(x=df$e, by=list(stocks_index), FUN=mean)$x
  denom <- aggregate(x=df$y, by=list(stocks_index), FUN=mean)$x
  return((1-mean(numer**2)/mean(denom**2))*100)
}

tf_residual = function(fit,Y,Z,H,months,no_H){
  # Tree Factor Models
  regressor = Z
  for(j in 1:dim(Z)[2])
  {
    regressor[,j] = Z[,j] * fit$ft[months + 1]
  }
  if(!no_H)
  {
    regressor = cbind(regressor, H)
  }
  # print(fit$R2*100)
  x <- as.matrix(regressor)
  y <- Y
  b_tf = solve(t(x)%*%x)%*%t(x)%*%y
  haty <- (x%*%b_tf)[,1]
  print(b_tf)
  return(Y-haty)
}


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


############################
### time series split  ###

print(dim(data))

######## insert here ########

######## insert here ########
xt = read.csv("../../../../data/xt_1972_2021_10y_stand.csv")
xt$X <- as.Date(xt$X, format='%Y/%m/%d')
xt_train = xt[xt$X >= start &  xt$X <= split, ]
head(xt_train)
######## insert here ########

######## insert here ########

write.csv(data.frame(train_ts_idx), paste0(case,"_train_ts_idx",".csv"))

sprintf("train proportion: %.2f \n", sum(train_ts_idx)/length(train_ts_idx))

# Char

all_chars <- names(data)[c(10:70)]
top5chars <- c(1:5)
instruments = all_chars[top5chars]
splitting_chars <- all_chars

first_split_var = c(1:61)-1
second_split_var = c(1:61)-1

first_split_var_boosting = c(1:61)-1
second_split_var_boosting = c(1:61)-1

##### train-test split #####

data[,c('date')] <- as.Date(data[,c('date')], format='%Y-%m-%d')
data1 <- data[(data[,c('date')]>=start) & (data[,c('date')]<=split), ]

###### train data for all boosting steps #####
X_train = data1[,splitting_chars]
R_train = data1[,c("xret")]
months_train = as.numeric(as.factor(data1[,c("date")]))
months_train = months_train - 1 # start from 0
stocks_train = as.numeric(as.factor(data1[,c("permno")])) - 1
Z_train = data1[, instruments]
Z_train = cbind(1, Z_train)
portfolio_weight_train = data1[,c("lag_me")]
loss_weight_train = data1[,c("lag_me")]
num_months = length(unique(months_train))
num_stocks = length(unique(stocks_train))

# write.csv(data.frame(stocks_train), paste0(case,"_stocks_train",".csv"))


# mkt
ff = read.csv('../../../../data/FactorsMonthly_202205.csv', row.names = 1)
names(ff) <- c('mktrf','smb','hml','rmw','cma','rf','mom')
ff <- ff/100
row.names(ff) <- as.Date(paste0(row.names(ff), '01'), format='%Y%m%d')

ff_train <- ff[row.names(ff)>=start,]
ff_train <- ff_train[row.names(ff_train)<=split,]
ff_train <- ff_train[train_ts_idx,]
avg_ff_train <- colMeans(ff_train)

mkt = as.matrix(ff_train[,c("mktrf")])

###### train data 1 #####
# the first H is the mkt
Y_train1 = data1[,c("xret")]
H_train1 = ff_train$mktrf

# train 1 
t = proc.time()

fit1 = PTree(R_train, Y_train1, X_train, Z_train, H_train1, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, num_stocks, 
num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
  no_H1, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

t = proc.time() - t
print(t)

# in sample check
insPred1 = predict(fit1, X_train, R_train, months_train, portfolio_weight_train)
sum((insPred1$ft - fit1$ft)^2)
print(fit1$R2)

############# START OUTPUT #############


tf_train <- cbind(ff_train$mktrf, fit1$ft)
avg_tf_train <- colMeans(tf_train)

# output factors
write.csv(data.frame(tf_train), paste0(case,"_bf_train",".csv"))

save(fit1, file = "fit1.RData")

############# END OUTPUT return and hat returns #############