
case = 'Oct'
# case_simu = NaN

data_src = paste0("../../../data/simu_kappa_",kappa,"_seed_",seq,".RData")

print(data_src)
load(data_src)


case=paste0("simu_",kappa,"_seed_",seq)
start = 0
split = 499
end = 999

max_depth=10
max_depth_boosting = 10

num_iter  = 9 
num_iterB = 9 

min_leaf_size = 20 

num_cutpoints = 4
equal_weight = FALSE
# no_H = TRUE
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

# re-order the data

data$lag_me = 1

tmp = data[,c('permno', 'date', 'lag_me',

               'me', 'bm', 'mom12m', 
        
              'ill', 'me_ia', 'chtx', 'mom36m', 're', 'depr', 'rd_sale', 'roa',
              'bm_ia', 'cfp', 'mom1m', 'baspread', 'rdm',  'sgr', 
              'std_dolvol', 'rvar_ff3', 'herf', 'sp', 'hire', 'pctacc',
              'grltnoa', 'turn', 'abr', 'seas1a', 'adm', 'cash', 'chpm',
              'cinvest', 'acc', 'gma', 'beta', 'sue', 'cashdebt', 'ep', 'lev',
              'op', 'alm', 'lgr', 'noa', 'roe', 'dolvol', 'rsup', 'std_turn',
              'maxret', 'mom6m', 'ni', 'nincr', 'ato', 'rna', 'agr', 'zerotrade',
              'chcsho', 'dy', 'rvar_capm', 'svar', 'mom60m', 'pscore', 'pm',
               
              'xret','mktrf')] 

print(dim(tmp))
data = tmp
rm(tmp)

# find mktrf
df_mkt <- as.data.frame(unique( data[ , c('mktrf','date') ] ))
df_mkt <- df_mkt[order(df_mkt[,c('date')]),]

# chars 

all_chars <- names(data)[c(4:64)]

top5chars <- c(1:61)
instruments = all_chars[top5chars]
splitting_chars <- all_chars

first_split_var = c(1:61)-1
second_split_var = c(1:61)-1

first_split_var_boosting = c(1:61)-1
second_split_var_boosting = c(1:61)-1


##### train-test split #####

# data[,c('date')] <- as.Date(data[,c('date')], format='%Y-%m-%d')
data1 <- data[(data[,c('date')]>=start) & (data[,c('date')]<=split), ]
data2 <- data[(data[,c('date')]> split) & (data[,c('date')]<=end), ]

mkt_train = df_mkt[ (df_mkt[,c('date')]>=start) & (df_mkt[,c('date')]<=split), ]
mkt_test  = df_mkt[ (df_mkt[,c('date')]> split) & (df_mkt[,c('date')]<=end), ]

rm(data)

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

###### train data 1 #####
# the first H is the mkt
Y_train1 = data1[,c("xret")]
H_train1 = mkt_train[,c('mktrf')]

# train 1 
t = proc.time()

fit1 = PTree(R_train, Y_train1, X_train, Z_train, H_train1, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, num_stocks, 
num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

t = proc.time() - t
print(t)

# in sample check
insPred1 = predict(fit1, X_train, R_train, months_train, portfolio_weight_train)
sum((insPred1$ft - fit1$ft)^2)
print(fit1$R2)

write.csv(data.frame(insPred1$leaf_index), paste0(case,"_leaf_index",".csv"))
write.csv(data.frame(months_train), paste0(case,"_months_train",".csv"))


###### train data 2 #####
# the first H is the mkt
Y_train2 = data1[,c("xret")]
H_train2 = cbind(mkt_train[,c('mktrf')],fit1$ft)

# train
fit2 = PTree(R_train, Y_train2, X_train, Z_train, H_train2, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 3 #####
# the first H is the mkt
Y_train3 = data1[,c("xret")]
H_train3 = cbind(mkt_train[,c('mktrf')],fit1$ft,fit2$ft)

# train
fit3 = PTree(R_train, Y_train3, X_train, Z_train, H_train3, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints,eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)


###### train data 4 #####
# the first H is the mkt
Y_train4 = data1[,c("xret")]
H_train4 = cbind(mkt_train[,c('mktrf')],fit1$ft,fit2$ft,fit3$ft)

# train
fit4 = PTree(R_train, Y_train4, X_train, Z_train, H_train4, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H,
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 5 #####
# the first H is the mkt
Y_train5 = data1[,c("xret")]
H_train5 = cbind(mkt_train[,c('mktrf')],fit1$ft,fit2$ft,fit3$ft,fit4$ft)

# train
fit5 = PTree(R_train, Y_train5, X_train, Z_train, H_train5, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf5 #############


###### train data 6 #####
# the first H is the mkt
Y_train6 = data1[,c("xret")]
H_train6 = cbind(mkt_train[,c('mktrf')],fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft)

# train
fit6 = PTree(R_train, Y_train6, X_train, Z_train, H_train6, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf6 #############


###### train data 7 #####
# the first H is the mkt
Y_train7 = data1[,c("xret")]
H_train7 = cbind(mkt_train[,c('mktrf')],fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft)

# train
fit7 = PTree(R_train, Y_train7, X_train, Z_train, H_train7, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf7 #############


###### train data 8 #####
# the first H is the mkt
Y_train8 = data1[,c("xret")]
H_train8 = cbind(mkt_train[,c('mktrf')],fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft)

# train
fit8 = PTree(R_train, Y_train8, X_train, Z_train, H_train8, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf8 #############


###### train data 9 #####
# the first H is the mkt
Y_train9 = data1[,c("xret")]
H_train9 = cbind(mkt_train[,c('mktrf')],fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft)

# train
fit9 = PTree(R_train, Y_train9, X_train, Z_train, H_train9, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf9 #############


###### train data 10 #####
# the first H is the mkt
Y_train10 = data1[,c("xret")]
H_train10 = cbind(mkt_train[,c('mktrf')],fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft)
 

# train
fit10 = PTree(R_train, Y_train10, X_train, Z_train, H_train10, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf10 #############


###### train data 11 #####
# the first H is the mkt
Y_train11 = data1[,c("xret")]
H_train11 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft
                  )

# train
fit11 = PTree(R_train, Y_train11, X_train, Z_train, H_train11, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf11 #############



###### train data 12 #####
# the first H is the mkt
Y_train12 = data1[,c("xret")]
H_train12 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,
                  fit11$ft
                  )

# train
fit12 = PTree(R_train, Y_train12, X_train, Z_train, H_train12, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf12 #############



###### train data 13 #####
# the first H is the mkt
Y_train13 = data1[,c("xret")]
H_train13 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,
                  fit11$ft,fit12$ft
                  )

# train
fit13 = PTree(R_train, Y_train13, X_train, Z_train, H_train13, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf13 #############


###### train data 14 #####
# the first H is the mkt
Y_train14 = data1[,c("xret")]
H_train14 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,
                  fit11$ft,fit12$ft,fit13$ft
                  )

# train
fit14 = PTree(R_train, Y_train14, X_train, Z_train, H_train14, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf14 #############



###### train data 15 #####
# the first H is the mkt
Y_train15 = data1[,c("xret")]
H_train15 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,
                  fit11$ft,fit12$ft,fit13$ft,fit14$ft
                  )

# train
fit15 = PTree(R_train, Y_train15, X_train, Z_train, H_train15, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf15 #############



###### train data 16 #####
# the first H is the mkt
Y_train16 = data1[,c("xret")]
H_train16 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,
                  fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft
                  )

# train
fit16 = PTree(R_train, Y_train16, X_train, Z_train, H_train16, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf16 #############


###### train data 17 #####
# the first H is the mkt
Y_train17 = data1[,c("xret")]
H_train17 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,
                  fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft,fit16$ft
                  )

# train
fit17 = PTree(R_train, Y_train17, X_train, Z_train, H_train17, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf17 #############


###### train data 18 #####
# the first H is the mkt
Y_train18 = data1[,c("xret")]
H_train18 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,
                  fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft,fit16$ft,fit17$ft
                  )

# train
fit18 = PTree(R_train, Y_train18, X_train, Z_train, H_train18, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf18 #############


###### train data 19 #####
# the first H is the mkt
Y_train19 = data1[,c("xret")]
H_train19 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,
                  fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft,fit16$ft,fit17$ft,fit18$ft
                  )

# train
fit19 = PTree(R_train, Y_train19, X_train, Z_train, H_train19, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf19 #############



###### train data 20 #####
# the first H is the mkt
Y_train20 = data1[,c("xret")]
H_train20 = cbind(mkt_train[,c('mktrf')],
                  fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,
                  fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft,fit16$ft,fit17$ft,fit18$ft,fit19$ft
                  )

# train
fit20 = PTree(R_train, Y_train20, X_train, Z_train, H_train20, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var_boosting, second_split_var_boosting, num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor,  early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

############# Train Period tf20 #############



###### test data #####

X_test = data2[,splitting_chars]
R_test = data2[,c("xret")]
months_test = as.numeric(as.factor(data2[,c("date")]))
months_test = months_test - 1 # start from 0
stocks_test = as.numeric(as.factor(data2[,c("permno")])) - 1
Z_test = data2[,instruments]
Z_test = cbind(1, Z_test)
# Z_test = 0
# H_test1 = c(1:(split+1))
portfolio_weight_test = data2[,c("lag_me")]
loss_weight_test = data2[,c("lag_me")]

############# Test Period Factors #############

pred1 = predict(fit1, X_test, R_test, months_test, portfolio_weight_test)
pred2 = predict(fit2, X_test, R_test, months_test, portfolio_weight_test)
pred3 = predict(fit3, X_test, R_test, months_test, portfolio_weight_test)
pred4 = predict(fit4, X_test, R_test, months_test, portfolio_weight_test)
pred5 = predict(fit5, X_test, R_test, months_test, portfolio_weight_test)

pred6 = predict(fit6, X_test, R_test, months_test, portfolio_weight_test)
pred7 = predict(fit7, X_test, R_test, months_test, portfolio_weight_test)
pred8 = predict(fit8, X_test, R_test, months_test, portfolio_weight_test)
pred9 = predict(fit9, X_test, R_test, months_test, portfolio_weight_test)
pred10 = predict(fit10, X_test, R_test, months_test, portfolio_weight_test)

pred11 = predict(fit11, X_test, R_test, months_test, portfolio_weight_test)
pred12 = predict(fit12, X_test, R_test, months_test, portfolio_weight_test)
pred13 = predict(fit13, X_test, R_test, months_test, portfolio_weight_test)
pred14 = predict(fit14, X_test, R_test, months_test, portfolio_weight_test)
pred15 = predict(fit15, X_test, R_test, months_test, portfolio_weight_test)

pred16 = predict(fit16, X_test, R_test, months_test, portfolio_weight_test)
pred17 = predict(fit17, X_test, R_test, months_test, portfolio_weight_test)
pred18 = predict(fit18, X_test, R_test, months_test, portfolio_weight_test)
pred19 = predict(fit19, X_test, R_test, months_test, portfolio_weight_test)
pred20 = predict(fit20, X_test, R_test, months_test, portfolio_weight_test)

############# START OUTPUT #############


tf_train <- cbind(mkt_train[ ,c('mktrf')],
                  fit1$ft, fit2$ft, fit3$ft, fit4$ft, fit5$ft,
                  fit6$ft, fit7$ft, fit8$ft, fit9$ft, fit10$ft,
                  fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft,
                  fit16$ft,fit17$ft,fit18$ft,fit19$ft,fit20$ft
                  )

tf_test  <- cbind(mkt_test[ ,c('mktrf')],
                  pred1$ft, pred2$ft, pred3$ft, pred4$ft, pred5$ft,
                  pred6$ft, pred7$ft, pred8$ft, pred9$ft, pred10$ft,
                  pred11$ft, pred12$ft, pred13$ft, pred14$ft, pred15$ft,
                  pred16$ft, pred17$ft, pred18$ft, pred19$ft, pred20$ft
                  )

# # output factors
write.csv(data.frame(tf_train), paste0(case,"_bf_train",".csv"))
write.csv(data.frame(tf_test), paste0(case,"_bf_test",".csv"))

write.csv(data.frame(insPred1$leaf_index), paste0(case,"_leaf_index",".csv"))
write.csv(data.frame(months_train), paste0(case,"_months_train",".csv"))
write.csv(data.frame(R_train), paste0(case,"_R_train",".csv"))
write.csv(data.frame(portfolio_weight_train), paste0(case,"_weight_train",".csv"))

write.csv(data.frame(fit1$leaf_weight), paste0(case,"_leaf_weight1",".csv"))
write.csv(data.frame(fit2$leaf_weight), paste0(case,"_leaf_weight2",".csv"))
write.csv(data.frame(fit3$leaf_weight), paste0(case,"_leaf_weight3",".csv"))
write.csv(data.frame(fit4$leaf_weight), paste0(case,"_leaf_weight4",".csv"))
write.csv(data.frame(fit5$leaf_weight), paste0(case,"_leaf_weight5",".csv"))

write.csv(data.frame(fit6$leaf_weight), paste0(case,"_leaf_weight6",".csv"))
write.csv(data.frame(fit7$leaf_weight), paste0(case,"_leaf_weight7",".csv"))
write.csv(data.frame(fit8$leaf_weight), paste0(case,"_leaf_weight8",".csv"))
write.csv(data.frame(fit9$leaf_weight), paste0(case,"_leaf_weight9",".csv"))
write.csv(data.frame(fit10$leaf_weight), paste0(case,"_leaf_weight10",".csv"))

write.csv(data.frame(fit11$leaf_weight), paste0(case,"_leaf_weight11",".csv"))
write.csv(data.frame(fit12$leaf_weight), paste0(case,"_leaf_weight12",".csv"))
write.csv(data.frame(fit13$leaf_weight), paste0(case,"_leaf_weight13",".csv"))
write.csv(data.frame(fit14$leaf_weight), paste0(case,"_leaf_weight14",".csv"))
write.csv(data.frame(fit15$leaf_weight), paste0(case,"_leaf_weight15",".csv"))

write.csv(data.frame(fit16$leaf_weight), paste0(case,"_leaf_weight16",".csv"))
write.csv(data.frame(fit17$leaf_weight), paste0(case,"_leaf_weight17",".csv"))
write.csv(data.frame(fit18$leaf_weight), paste0(case,"_leaf_weight18",".csv"))
write.csv(data.frame(fit19$leaf_weight), paste0(case,"_leaf_weight19",".csv"))
write.csv(data.frame(fit20$leaf_weight), paste0(case,"_leaf_weight20",".csv"))

### output basis portfolio returns

write.csv(data.frame(fit1$portfolio), paste0(case,"_portfolio_fit1",".csv"))
write.csv(data.frame(pred1$portfolio), paste0(case,"_portfolio_pred1",".csv"))

write.csv(data.frame(fit2$portfolio), paste0(case,"_portfolio_fit2",".csv"))
write.csv(data.frame(pred2$portfolio), paste0(case,"_portfolio_pred2",".csv"))

write.csv(data.frame(fit3$portfolio), paste0(case,"_portfolio_fit3",".csv"))
write.csv(data.frame(pred3$portfolio), paste0(case,"_portfolio_pred3",".csv"))

write.csv(data.frame(fit4$portfolio), paste0(case,"_portfolio_fit4",".csv"))
write.csv(data.frame(pred4$portfolio), paste0(case,"_portfolio_pred4",".csv"))

write.csv(data.frame(fit5$portfolio), paste0(case,"_portfolio_fit5",".csv"))
write.csv(data.frame(pred5$portfolio), paste0(case,"_portfolio_pred5",".csv"))

write.csv(data.frame(fit6$portfolio), paste0(case,"_portfolio_fit6",".csv"))
write.csv(data.frame(pred6$portfolio), paste0(case,"_portfolio_pred6",".csv"))

write.csv(data.frame(fit7$portfolio), paste0(case,"_portfolio_fit7",".csv"))
write.csv(data.frame(pred7$portfolio), paste0(case,"_portfolio_pred7",".csv"))

write.csv(data.frame(fit8$portfolio), paste0(case,"_portfolio_fit8",".csv"))
write.csv(data.frame(pred8$portfolio), paste0(case,"_portfolio_pred8",".csv"))

write.csv(data.frame(fit9$portfolio), paste0(case,"_portfolio_fit9",".csv"))
write.csv(data.frame(pred9$portfolio), paste0(case,"_portfolio_pred9",".csv"))

write.csv(data.frame(fit10$portfolio), paste0(case,"_portfolio_fit10",".csv"))
write.csv(data.frame(pred10$portfolio), paste0(case,"_portfolio_pred10",".csv"))


write.csv(data.frame(fit11$portfolio), paste0(case,"_portfolio_fit11",".csv"))
write.csv(data.frame(pred11$portfolio), paste0(case,"_portfolio_pred11",".csv"))

write.csv(data.frame(fit12$portfolio), paste0(case,"_portfolio_fit12",".csv"))
write.csv(data.frame(pred12$portfolio), paste0(case,"_portfolio_pred12",".csv"))

write.csv(data.frame(fit13$portfolio), paste0(case,"_portfolio_fit13",".csv"))
write.csv(data.frame(pred13$portfolio), paste0(case,"_portfolio_pred13",".csv"))

write.csv(data.frame(fit14$portfolio), paste0(case,"_portfolio_fit14",".csv"))
write.csv(data.frame(pred14$portfolio), paste0(case,"_portfolio_pred14",".csv"))

write.csv(data.frame(fit15$portfolio), paste0(case,"_portfolio_fit15",".csv"))
write.csv(data.frame(pred15$portfolio), paste0(case,"_portfolio_pred15",".csv"))

write.csv(data.frame(fit16$portfolio), paste0(case,"_portfolio_fit16",".csv"))
write.csv(data.frame(pred16$portfolio), paste0(case,"_portfolio_pred16",".csv"))

write.csv(data.frame(fit17$portfolio), paste0(case,"_portfolio_fit17",".csv"))
write.csv(data.frame(pred17$portfolio), paste0(case,"_portfolio_pred17",".csv"))

write.csv(data.frame(fit18$portfolio), paste0(case,"_portfolio_fit18",".csv"))
write.csv(data.frame(pred18$portfolio), paste0(case,"_portfolio_pred18",".csv"))

write.csv(data.frame(fit19$portfolio), paste0(case,"_portfolio_fit19",".csv"))
write.csv(data.frame(pred19$portfolio), paste0(case,"_portfolio_pred19",".csv"))

write.csv(data.frame(fit20$portfolio), paste0(case,"_portfolio_fit20",".csv"))
write.csv(data.frame(pred20$portfolio), paste0(case,"_portfolio_pred20",".csv"))


save(fit1, file = "fit1.RData")
save(fit2, file = "fit2.RData")
save(fit3, file = "fit3.RData")
save(fit4, file = "fit4.RData")
save(fit5, file = "fit5.RData")

save(fit6, file = "fit6.RData")
save(fit7, file = "fit7.RData")
save(fit8, file = "fit8.RData")
save(fit9, file = "fit9.RData")
save(fit10, file = "fit10.RData")

save(fit11, file = "fit11.RData")
save(fit12, file = "fit12.RData")
save(fit13, file = "fit13.RData")
save(fit14, file = "fit14.RData")
save(fit15, file = "fit15.RData")

save(fit16, file = "fit16.RData")
save(fit17, file = "fit17.RData")
save(fit18, file = "fit18.RData")
save(fit19, file = "fit19.RData")
save(fit20, file = "fit20.RData")

############# END OUTPUT #############