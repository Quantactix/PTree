

print("Length random chars: \n")
print(length(random_chars))
print("Length random dates: \n")
print(length(random_dates))

write.csv(data.frame(random_chars), paste0('./output_',case,"/random_chars.csv"))
write.csv(data.frame(random_dates), paste0('./output_',case,"/random_dates.csv"))

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
end   = '2020-12-31'

min_leaf_size = 20
max_depth = 3
num_iter = 1000
num_cutpoints = 4
equal_weight = FALSE
no_H = TRUE
abs_normalize = TRUE 
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

range_constraint = FALSE
upper_limit = Inf
lower_limit = 0
gamma_optimize = 2.0
sum_constraint = 1.0

a1=0.0
a2=0.0
list_K = matrix(rep(0,3), nrow = 3, ncol = 1)

random_split = FALSE

##### load data #####

load("../../data/data_202312.rda")

# re-order the data

tmp = data[,c('gvkey', 'permno', 'sic', 'ret', 'exchcd', 'shrcd', 'date', 'ffi49',
              'lag_me', 
              # 10
              'rank_me', 'rank_bm', 'rank_agr', 'rank_op', 'rank_mom12m',
              
              # 15
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
              'rank_pctacc', 'rank_chtx', # 70-1=69
              
              'log_me', # 71-1=70
              
              'MKTRF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', # 77-1=76,  FF5 
              
#               "Q4MKT", "Q4ME", "Q4IA", "Q4ROE", # 81, Q4

#               # "rp5_1","rp5_2","rp5_3", # 84
              # "rp10_1","rp10_2","rp10_3", # 87 # 84
#               # "rp20_1","rp20_2","rp20_3", #90 # 84

              # "ip_1","ip_2","ip_3", # 87
              
              'x_tbl', 'x_dfy', 'x_tms', 'x_svar', 'x_ill', 'x_infl',
              'x_dy', 'x_lev', 'x_ep', 'x_ni', # 97, x_{t-1}  # 87-1=86
               
              'xret')] # 98, excess return  # 88-1=87

data = tmp
rm(tmp)

# chars 

all_chars <- names(data)[c(10:70)]
top5chars <- c(1:5)
instruments = all_chars[top5chars]
splitting_chars <- all_chars[random_chars]
# output the splitting_chars
write.csv(data.frame(splitting_chars), paste0('./output_',case,"/splitting_chars.csv"))

first_split_var = c(1:20)-1
second_split_var = c(1:20)-1

##### train-test split #####

data[,c('date')] <- as.Date(data[,c('date')], format='%Y-%m-%d')
data1 <- data[(data[,c('date')]>=start) & (data[,c('date')]<=split), ]
data2 <- data[(data[,c('date')]>split) & (data[,c('date')]<=end), ]

rm(data)
print("Finish Read Data")

# ###### train data FOR MKT BETA #####

X_train = data1[,splitting_chars]
Y_train = data1[,c("xret")]
months_train = as.numeric(as.factor(data1[,c("date")]))
months_train = months_train - 1 # start from 0
stocks_train = as.numeric(as.factor(data1[,c("permno")])) - 1
Z_train = data1[, instruments]
Z_train = cbind(1, Z_train)
H_train = data1[,c("MKTRF")]
H_train = H_train * Z_train
portfolio_weight_train = data1[,c("lag_me")]
loss_weight_train = data1[,c("lag_me")]

### first run mkt with original sample ###

# mkt
ff = read.csv('../../data/FactorsMonthly_202312.csv', row.names = 1)
names(ff) <- c('mktrf','smb','hml','rmw','cma','rf','mom')
ff <- ff/100
row.names(ff) <- as.Date(paste0(row.names(ff), '01'), format='%Y%m%d')

ff_train <- ff[row.names(ff)>=start,]
ff_train <- ff_train[row.names(ff_train)<=split,]
avg_ff_train <- colMeans(ff_train)

ff_test <- ff[row.names(ff)>split,]
ff_test <- ff_test[row.names(ff_test)<=end,]
avg_ff_test <- colMeans(ff_test)



###### bootstrap train data #####

# bootstrap
all_dates <- unique(data1[,c("date")])
all_dates <- sort(all_dates)
data3 <- data.frame()
data31 <- data.frame()
data32 <- data.frame()

print("Length of all_dates: \n")
print(length(all_dates))

t1 = proc.time()

for (i in c(1:(length(all_dates)/2))){
  d <- all_dates[i]
  bsd <- all_dates[random_dates[i]]
  # print("\n")
  # print(i)
  # print(d)
  # print("\n")
  # print(bsd)
  tmp <- data1[data1[,c("date")]==bsd,]
  tmp[,c("date")] <- d
  data31 <- rbind(data31, tmp)
}

t2 = proc.time()

for (i in c((length(all_dates)/2+1):length(all_dates))){
  d <- all_dates[i]
  bsd <- all_dates[random_dates[i]]
  # print("\n")
  # print(i)
  # print(d)
  # print("\n")
  # print(bsd)
  tmp <- data1[data1[,c("date")]==bsd,]
  tmp[,c("date")] <- d
  data32 <- rbind(data32, tmp)
}

t3 = proc.time()

data3 <- rbind(data31, data32)

t4 = proc.time()

rm(data31, data32)


# repeat train-data steps
X_bs = data3[,splitting_chars]
R_bs = data3[,c("xret")]
# Y_bs = data3[,c("res1")]
months_bs = as.numeric(as.factor(data3[,c("date")]))
months_bs = months_bs - 1 # start from 0
stocks_bs = as.numeric(as.factor(data3[,c("permno")])) - 1
Z_bs = data3[, instruments]
Z_bs = cbind(1, Z_bs)
# H_bs = data3[,c("MKTRF")]
# H_bs = H_bs * Z_bs
portfolio_weight_bs = data3[,c("lag_me")]
loss_weight_bs = data3[,c("lag_me")]

Y_bs = data3[,c("xret")]
H_bs = ff_train$mktrf[random_dates]

rm(data3)
print("Finish Bootstrap Data")

##### bootstrap sample train #####

num_months = length(unique(months_bs))
num_stocks = length(unique(stocks_bs))

t = proc.time()
# boostrap sample
fit = TreeFactor_APTree(R_bs, Y_bs, X_bs, Z_bs, H_bs, portfolio_weight_bs, 
loss_weight_bs, stocks_bs, months_bs, first_split_var, second_split_var, num_stocks, 
num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)
t = proc.time() - t

print(t)


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

#### output the portfolios and sdf #####

write.csv(data.frame(fit$leaf_weight), paste0('./output_',case,"/train_weight.csv"))

# ins
write.csv(data.frame(fit$portfolio), paste0('./output_',case,"/train_portfolio.csv"))
write.csv(data.frame(fit$ft), paste0('./output_',case,"/train_ft.csv"))

print("Finish All.")