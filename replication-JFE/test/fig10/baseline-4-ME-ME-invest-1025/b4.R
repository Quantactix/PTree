

case = paste0('char_',i)
loopChar = c(1,i)
print(loopChar)

###### library #####

library(TreeFactor)
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


myOLS<-function(X,Y){
  dim(X)
  dim(Y)
  
  X1 = cbind(1,X)
  dim(X1)
  
  b = solve( t(X1) %*% X1 ) %*% t(X1) %*% Y
  b
  
  e = Y-X1%*%b
  d = dim(X1)
  MSE = sum(e**2)/(d[1]-d[2])
  var_b = MSE * diag( solve( t(X1) %*% X1 ) )
  sd_b = var_b**(0.5)
  
  t_b = b/sd_b
  t_b
  
  return (list(b[1],t_b[1]))
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

g_residual = function(g,Y,Z,H,months,no_H){
  # Tree Factor Models
  regressor = Z
  for(j in 1:dim(Z)[2])
  {
    regressor[,j] = Z[,j] * g[months + 1]
  }
  if(!no_H)
  {
    regressor = cbind(regressor, H)
  }
  # print(fit$R2*100)
  x <- as.matrix(regressor)
  y <- Y
  b_g = solve(t(x)%*%x)%*%t(x)%*%y
  haty <- (x%*%b_g)[,1]
  print(b_g)
  return(Y-haty)
}

###### parameters #####

start = '1981-01-01'
split = '2020-12-31'
# end   = '2020-12-31'

min_leaf_size = 10

max_depth = 4
num_iter = 4

num_cutpoints = 4
equal_weight = FALSE
no_H1 = TRUE
no_H = FALSE
abs_normalize = FALSE
weighted_loss = FALSE
# stop_no_gain = FALSE

early_stop = FALSE
stop_threshold = 1
lambda_ridge = 0

#   this tiny regularization ensures the matrix inversion
# penalty for the sigma (sigma + lambda I)^{-1} * mu
lambda_mean = 0
lambda_cov = 1e-4

lambda_mean_factor = 0
lambda_cov_factor = 1e-5

eta=1

a1=0.0
a2=0.0
list_K = matrix(rep(0,3), nrow = 3, ncol = 1)

##### load data #####

library(arrow)
my_tibble = arrow::read_feather("../../data/data_202312.feather")
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
splitting_chars <- all_chars[loopChar]
splitting_chars_base <- all_chars[c(1)]   # ME 
# splitting_chars_base <- all_chars[c(1:2)] # ME BM

print("splitting_chars \n")
print(splitting_chars)

print("splitting_chars_base \n")
print(splitting_chars_base)

first_split_var_boosting = c(0)   # the first  depth split
second_split_var_boosting = c(0)  # the second depth split

##### train-test split #####

data[,c('date')] <- as.Date(data[,c('date')], format='%Y-%m-%d')
# data1 <- data[(data[,c('date')]>=start) & (data[,c('date')]<=split), ]
# data2 <- data[(data[,c('date')]>split) & (data[,c('date')]<=end), ]

data1 <- data

rm(data)
print("Finish Read Data")

###### train data for all boosting steps #####
X_train = data1[,splitting_chars]
X_train_base = data1[,splitting_chars_base]
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

### mkt ###

Y_train = data1[,c("xret")]
H_train = data1[,c("MKTRF")]

# mkt
ff = read.csv('../../data/FactorsMonthly_202205.csv', row.names = 1)
names(ff) <- c('mktrf','smb','hml','rmw','cma','rf','mom')
ff <- ff/100
row.names(ff) <- as.Date(paste0(row.names(ff), '01'), format='%Y%m%d')

ff_train <- ff[row.names(ff)>=start,]
ff_train <- ff_train[row.names(ff_train)<=split,]
avg_ff_train <- colMeans(ff_train)

Y_train1 = data1[,c("xret")]
H_train1 = ff_train$mktrf

# ### Baseline PTree ###

# # train 1 
# t = proc.time()

# fit_base = TreeFactor_APTree(R_train, Y_train1, X_train_base, Z_train, H_train1, portfolio_weight_train, 
# loss_weight_train, stocks_train, months_train, 
#   first_split_var_boosting, 
#   second_split_var_boosting, 
# num_stocks, 
# num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
#   no_H1, 
# abs_normalize, weighted_loss, 
# lambda_mean, lambda_cov, 
# lambda_mean_factor, lambda_cov_factor, 
# early_stop, stop_threshold, lambda_ridge, a1, a2, list_K)

# t = proc.time() - t
# print(t)

# my_string1 <- fit_base$tree
# my_string_split1 <- scan(text = my_string1, what = "")
# fit1_char1 <- as.numeric(my_string_split1[3])+1

# print("\n #### fit1_char1 #### \n")
# print(fit1_char1)
# print(all_chars[fit1_char1])
# print("\n #### fit1_char1 #### \n")

# criteria_base = min(unlist(fit_base$all_criterion[3]))
# print("Criteria Base Start \n")
# print("\n")
# print(criteria_base)
# print("Criteria Base End \n")


### Expanded Baseline PTree ###

t = proc.time()
# boostrap sample
fit = TreeFactor_APTree(R_train, Y_train1, X_train, Z_train, H_train1, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, 
  first_split_var_boosting,
  second_split_var_boosting, 
num_stocks, 
num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
no_H1, abs_normalize, weighted_loss, 
lambda_mean, lambda_cov,
lambda_mean_factor, lambda_cov_factor,
early_stop, stop_threshold, lambda_ridge,
a1, a2, list_K)

t = proc.time() - t

print(t)

# print R2 #

criteria_char = min(unlist(fit$all_criterion[3]))
print("Criteria Char Start \n")
print("\n")
print(criteria_char)
print("Criteria Char End \n")

# sample check

insPred = predict(fit, X_train, R_train, months_train, portfolio_weight_train)
sum((insPred$ft - fit$ft)^2)

# print R2 #

# print("Criteria (1 - Char/Base)*100 Start \n")
# print("\n")
# print( (1-criteria_char/criteria_base)*100 )
# print("Criteria (1 - Char/Base)*100 End \n")

##### Print Tree Structure #####

nodes <- strsplit(fit$tree, '\n')[[1]]
list_chars = rep(NaN,length(nodes))
list_nodes = rep(NaN,length(nodes))
list_seq = rep(NaN,length(nodes))
for (i in c(2:length(nodes))){
  print(i)
  n <- nodes[i]
  print(n)
  node_id <- as.numeric(strsplit(n,' ')[[1]][1])
  list_nodes[i] <- node_id
  seqi <- as.numeric(strsplit(n,' ')[[1]][5])
  list_seq[i] <- seqi
  if ( (as.numeric(strsplit(n,' ')[[1]][2])==0) & ((as.numeric(strsplit(n,' ')[[1]][3])==0)) & ((as.numeric(strsplit(n,' ')[[1]][4])==0)) ){
    charid <- NaN
  }else{
    charid <- as.numeric(strsplit(n,' ')[[1]][2])+1
  }
  list_chars[i] <- charid
}


print("Split Sequence: ")
print(list_seq)

print("Split Nodes: ")
print(list_nodes)

print("Split Chars: ")
print(names(X_train)[list_chars])

print("Split Chars in Depth 1 and 2: ")
key_chars <- names(X_train)[list_chars[match(c(1:7),list_nodes)]]
print(key_chars)

print("Number of leaves: ")
print(length(fit$leaf_id))

print("All Chars: ")
print(names(X_train)[c(list_chars)])

print("Unique chars: ")
uniqe_chars <- names(X_train)[unique(list_chars)]
print(length(uniqe_chars))
print(uniqe_chars)
print("Finish All.")
