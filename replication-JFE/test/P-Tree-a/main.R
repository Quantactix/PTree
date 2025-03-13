t = proc.time()

###### parameters #####

case='yr_1981_2020_num_iter_9_boost_20' 

start = '1981-01-01' 
split = '2020-12-31' 
end = '2020-12-31' 

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
splitting_chars <- all_chars

first_split_var = c(1:61)-1
second_split_var = c(1:61)-1

##### train-test split #####

data[,c('date')] <- as.Date(data[,c('date')], format='%Y-%m-%d')
data1 <- data[(data[,c('date')]>=start) & (data[,c('date')]<=split), ]

###### train data for all boosting steps #####
X_train = data1[,splitting_chars]
R_train = data1[,c("xret")]
months_train = as.numeric(as.factor(data1[,c("date")]))
months_train = months_train - 1 
stocks_train = as.numeric(as.factor(data1[,c("permno")])) - 1
Z_train = data1[, instruments]
Z_train = cbind(1, Z_train)

portfolio_weight_train = data1[,c("lag_me")]
loss_weight_train = data1[,c("lag_me")]
num_months = length(unique(months_train))
num_stocks = length(unique(stocks_train))


###### BENCHMARK FACTORS ##### 

# mkt
ff = read.csv('../../data/FactorsMonthly_202312.csv', row.names = 1)
names(ff) <- c('mktrf','smb','hml','rmw','cma','rf','mom')
ff <- ff/100
row.names(ff) <- as.Date(paste0(row.names(ff), '01'), format='%Y%m%d')

ff_train <- ff[row.names(ff)>=start,]
ff_train <- ff_train[row.names(ff_train)<=split,]
avg_ff_train <- colMeans(ff_train)

###### train data 1 #####
# the first `H`` is the `mkt``, but we use `no_H1 = TRUE`, so the algorithm ignores the `H`.
Y_train1 = data1[,c("xret")]
H_train1 = ff_train$mktrf

# train 1 
t1 = proc.time()

fit1 = PTree(R_train, Y_train1, X_train, Z_train, H_train1, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, num_stocks, 
num_months, min_leaf_size, max_depth, num_iter, num_cutpoints, eta, equal_weight, 
  no_H1, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

t1 = proc.time() - t1
print(t1)

###### train data 2 #####
# the second `H`` is the firts P-Tree factor, use `no_H = FALSE`, so the algprithm use the first P-Tree factor as benchmark.
Y_train2 = data1[,c("xret")]
H_train2 = fit1$ft

# train
fit2 = PTree(R_train, Y_train2, X_train, Z_train, H_train2, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
  num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 3 #####
Y_train3 = data1[,c("xret")]
H_train3 = cbind(fit1$ft,fit2$ft)

# train
fit3 = PTree(R_train, Y_train3, X_train, Z_train, H_train3, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints,eta, equal_weight, 
no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 4 #####
Y_train4 = data1[,c("xret")]
H_train4 = cbind(fit1$ft,fit2$ft,fit3$ft)

# train
fit4 = PTree(R_train, Y_train4, X_train, Z_train, H_train4, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H,
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 5 #####
Y_train5 = data1[,c("xret")]
H_train5 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft)

# train
fit5 = PTree(R_train, Y_train5, X_train, Z_train, H_train5, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 6 #####
Y_train6 = data1[,c("xret")]
H_train6 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft)

# train
fit6 = PTree(R_train, Y_train6, X_train, Z_train, H_train6, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 7 #####
Y_train7 = data1[,c("xret")]
H_train7 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft)

# train
fit7 = PTree(R_train, Y_train7, X_train, Z_train, H_train7, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 8 #####
Y_train8 = data1[,c("xret")]
H_train8 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft)

# train
fit8 = PTree(R_train, Y_train8, X_train, Z_train, H_train8, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 9 #####
Y_train9 = data1[,c("xret")]
H_train9 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft)

# train
fit9 = PTree(R_train, Y_train9, X_train, Z_train, H_train9, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 10 #####
Y_train10 = data1[,c("xret")]
H_train10 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft)

# train
fit10 = PTree(R_train, Y_train10, X_train, Z_train, H_train10, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 11 #####
Y_train11 = data1[,c("xret")]
H_train11 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft)

# train
fit11 = PTree(R_train, Y_train11, X_train, Z_train, H_train11, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 12 #####
Y_train12 = data1[,c("xret")]
H_train12 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,fit11$ft)

# train
fit12 = PTree(R_train, Y_train12, X_train, Z_train, H_train12, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 13 #####
Y_train13 = data1[,c("xret")]
H_train13 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,fit10$ft,fit11$ft,fit12$ft)

# train
fit13 = PTree(R_train, Y_train13, X_train, Z_train, H_train13, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

my_string13 <- fit13$tree
my_string_split13 <- scan(text = my_string13, what = "")
fit13_char1 <- as.numeric(my_string_split13[3])+1

###### train data 14 #####
Y_train14 = data1[,c("xret")]
H_train14 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,
                  fit10$ft,fit11$ft,fit12$ft,fit13$ft
                  )

# train
fit14 = PTree(R_train, Y_train14, X_train, Z_train, H_train14, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 15 #####
Y_train15 = data1[,c("xret")]
H_train15 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,
                  fit10$ft,fit11$ft,fit12$ft,fit13$ft,fit14$ft
                  )

# train
fit15 = PTree(R_train, Y_train15, X_train, Z_train, H_train15, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 16 #####
Y_train16 = data1[,c("xret")]
H_train16 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,
                  fit10$ft,fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft
                  )

# train
fit16 = PTree(R_train, Y_train16, X_train, Z_train, H_train16, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, 
num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 17 #####
Y_train17 = data1[,c("xret")]
H_train17 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,
                  fit10$ft,fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft,fit16$ft
                  )

# train
fit17 = PTree(R_train, Y_train17, X_train, Z_train, H_train17, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 18 #####
Y_train18 = data1[,c("xret")]
H_train18 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,
                  fit10$ft,fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft,fit16$ft,fit17$ft
                  )

# train
fit18 = PTree(R_train, Y_train18, X_train, Z_train, H_train18, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var, second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 19 #####
Y_train19 = data1[,c("xret")]
H_train19 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,
                  fit10$ft,fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft,fit16$ft,fit17$ft,fit18$ft
                  )

# train
fit19 = PTree(R_train, Y_train19, X_train, Z_train, H_train19, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var,  second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)

###### train data 20 #####
Y_train20 = data1[,c("xret")]
H_train20 = cbind(fit1$ft,fit2$ft,fit3$ft,fit4$ft,fit5$ft,fit6$ft,fit7$ft,fit8$ft,fit9$ft,
                  fit10$ft,fit11$ft,fit12$ft,fit13$ft,fit14$ft,fit15$ft,fit16$ft,fit17$ft,fit18$ft,fit19$ft
                  )

# train
fit20 = PTree(R_train, Y_train20, X_train, Z_train, H_train20, portfolio_weight_train, 
loss_weight_train, stocks_train, months_train, first_split_var,  second_split_var, 
num_stocks, num_months, min_leaf_size, max_depth_boosting, num_iterB, num_cutpoints, eta, equal_weight, 
  no_H, 
abs_normalize, weighted_loss, lambda_mean, lambda_cov, lambda_mean_factor, lambda_cov_factor, early_stop, stop_threshold, lambda_ridge, a1, a2, list_K, random_split)


############# OUTPUT Main #############

tf_train <- cbind(fit1$ft, fit2$ft, fit3$ft, fit4$ft, fit5$ft,
                  fit6$ft, fit7$ft, fit8$ft, fit9$ft, fit10$ft,
                  fit11$ft, fit12$ft, fit13$ft, fit14$ft, fit15$ft,
                  fit16$ft, fit17$ft, fit18$ft, fit19$ft, fit20$ft
                  )

write.csv(data.frame(tf_train), paste0(case,"_bf_train",".csv"))

t = proc.time() - t
print(t)

############# OUTPUT Others #############

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


## for ptree plot
write.csv(data.frame(months_train), paste0(case,"_months_train",".csv"))

insPred1 = predict(fit1, X_train, R_train, months_train, portfolio_weight_train)
write.csv(data.frame(insPred1$leaf_index), paste0(case,"_leaf_index1",".csv"))

write.csv(data.frame(fit1$leaf_weight), paste0(case,"_leaf_weight1",".csv"))

write.csv(data.frame(fit1$portfolio), paste0(case,"_portfolio_fit1",".csv"))
write.csv(data.frame(fit2$portfolio), paste0(case,"_portfolio_fit2",".csv"))
write.csv(data.frame(fit3$portfolio), paste0(case,"_portfolio_fit3",".csv"))
write.csv(data.frame(fit4$portfolio), paste0(case,"_portfolio_fit4",".csv"))
write.csv(data.frame(fit5$portfolio), paste0(case,"_portfolio_fit5",".csv"))

write.csv(data.frame(fit6$portfolio), paste0(case,"_portfolio_fit6",".csv"))
write.csv(data.frame(fit7$portfolio), paste0(case,"_portfolio_fit7",".csv"))
write.csv(data.frame(fit8$portfolio), paste0(case,"_portfolio_fit8",".csv"))
write.csv(data.frame(fit9$portfolio), paste0(case,"_portfolio_fit9",".csv"))
write.csv(data.frame(fit10$portfolio), paste0(case,"_portfolio_fit10",".csv"))

write.csv(data.frame(fit11$portfolio), paste0(case,"_portfolio_fit11",".csv"))
write.csv(data.frame(fit12$portfolio), paste0(case,"_portfolio_fit12",".csv"))
write.csv(data.frame(fit13$portfolio), paste0(case,"_portfolio_fit13",".csv"))
write.csv(data.frame(fit14$portfolio), paste0(case,"_portfolio_fit14",".csv"))
write.csv(data.frame(fit15$portfolio), paste0(case,"_portfolio_fit15",".csv"))

write.csv(data.frame(fit16$portfolio), paste0(case,"_portfolio_fit16",".csv"))
write.csv(data.frame(fit17$portfolio), paste0(case,"_portfolio_fit17",".csv"))
write.csv(data.frame(fit18$portfolio), paste0(case,"_portfolio_fit18",".csv"))
write.csv(data.frame(fit19$portfolio), paste0(case,"_portfolio_fit19",".csv"))
write.csv(data.frame(fit20$portfolio), paste0(case,"_portfolio_fit20",".csv"))
