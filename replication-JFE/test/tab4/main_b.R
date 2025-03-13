########## Set data loading path ##########
prepare_path = '../../data/'
output_path = 'performance/'
figure_path = 'figure/'

if (!dir.exists(output_path)){
  dir.create(output_path)
}

if (!dir.exists(figure_path)){
  dir.create(figure_path)
}

source("functions_b.R")
library("timeDate")      

########## Set Parameters ##########

case = "P-Tree-b"
criterion = 'mse' # or 'loss'

# Split dataset
start = '1981-01-01'
# split = '2000-12-31'
end = '2000-12-31'

ff5 = c('MKTRF','SMB','HML','RMW','CMA')
q5 = c("R_MKT","R_ME","R_IA","R_ROE","R_EG")
rppca = c('RPPCA_1','RPPCA_2','RPPCA_3','RPPCA_4','RPPCA_5')
ipca = c('IPCA_1','IPCA_2','IPCA_3','IPCA_4','IPCA_5')

pt_factor = c('PT1','PT2','PT3','PT4','PT5','PT6','PT7','PT8','PT9','PT10','PT11','PT12','PT13','PT14','PT15','PT16','PT17','PT18','PT19','PT20')
pt_factor_1_plusMKT = c('MKTRF','PT1')
pt_factor_1 = c('PT1')
pt_factor_5 = c('PT1','PT2','PT3','PT4','PT5')
pt_factor_10 = c('PT1','PT2','PT3','PT4','PT5','PT6','PT7','PT8','PT9','PT10')
pt_factor_15 = c('PT1','PT2','PT3','PT4','PT5','PT6','PT7','PT8','PT9','PT10','PT11','PT12','PT13','PT14','PT15')
pt_factor_20 = c('PT1','PT2','PT3','PT4','PT5','PT6','PT7','PT8','PT9','PT10','PT11','PT12','PT13','PT14','PT15','PT16','PT17','PT18','PT19','PT20')

# Load packages
library(stringr); library(MASS)

########## Data Preparation ##########

# read factor data
df_factor = read.csv("../../data/FactorsMonthly_202312.csv") / 100
colnames(df_factor) = c('X','MKTRF','SMB','HML','RMW','CMA','RF','MOM')
df_factor$date = seq(as.Date("1963/07/01"), as.Date("2022/05/01"),"months")
df_factor = df_factor[(df_factor['date']>=start) & (df_factor['date']<=end),]

df_q5 = read.csv("../../data/Q5/q5_factors_monthly_2021.csv")/100
df_q5$date = seq(as.Date("1967/01/01"), as.Date("2021/12/01"),"months")
df_q5 = df_q5[(df_q5['date']>=start) & (df_q5['date']<=end),]
df_q5 = df_q5[c("date","R_MKT","R_ME","R_IA","R_ROE","R_EG")]

df_rppca = read.csv("../../data/RP-PCA/rppca_ins_1981_2000.csv",header = FALSE); colnames(df_rppca) = c(rppca)

df_ipca = read.csv("../../data/IPCA/ipca_ins_1981_2000.csv"); colnames(df_ipca) = c("date", ipca)
df_ipca$date = seq(as.Date(start), as.Date(end),"months")

df_pt =  read.csv("../P-Tree-b/yr_1981_2000__2001_2020_num_iter_9_boost_20_bf_train.csv"); colnames(df_pt) = c("date", pt_factor)

# prepare dataframe for factor models
data_capm = as.matrix(df_factor[,c("MKTRF")])
data_ff5 = df_factor[,ff5]
data_q5 = df_q5[,q5]
data_rppca = df_rppca[,rppca]
data_ipca = df_ipca[,ipca]

data_pt_1 = as.matrix(df_pt[,c(pt_factor_1)])
data_pt_5 = df_pt[,pt_factor_5]
data_pt_10 = df_pt[,pt_factor_10]
data_pt_15 = df_pt[,pt_factor_15]
data_pt_20 = df_pt[,pt_factor_20]

##### Portfolio R2 #####

# Prepare for data set
rf = df_factor$RF

unisort <- read.csv("../../data/download_portfolios/Uni.csv");colnames(unisort)[1] <- "date";unisort <- data.frame(unisort, row.names = 1)
bisort <- read.csv("../../data/download_portfolios/Bi.csv");colnames(bisort)[1] <- "date";bisort <- data.frame(bisort, row.names = 1)
mebm25 <- read.csv("../../data/download_portfolios/mebm.CSV");colnames(mebm25)[1] <- "date";mebm25 <- data.frame(mebm25, row.names = 1)
ind49 <- read.csv("../../data/download_portfolios/ind49.CSV");colnames(ind49)[1] <- "date";ind49 <- data.frame(ind49, row.names = 1)

## these pt$x data need hand copy paste, which needs to be automated for publication.
pt1 <- read.csv("tmp/P-Tree-b/PT1.csv");colnames(pt1)[1] <- "date";pt1 <- data.frame(pt1, row.names = 1)
pt5 <- read.csv("tmp/P-Tree-b/PT_top5.csv");colnames(pt5)[1] <- "date";pt5 <- data.frame(pt5, row.names = 1)
pt10 <- read.csv("tmp/P-Tree-b/PT_top10.csv");colnames(pt10)[1] <- "date";pt10 <- data.frame(pt10, row.names = 1)
pt20 <- read.csv("tmp/P-Tree-b/PT_top20.csv");colnames(pt20)[1] <- "date";pt20 <- data.frame(pt20, row.names = 1)

# Start to calculate R2
port_r2 = c()

print('Start to calculate Portfolio R2: ')
g_list = c(); n_layer_list = c(); n_factor_list = c(); regularization_list = c(); beta_regular_list = c(); min_n_factor = c()
port_rtn_list = c("pt5","pt10","pt20","unisort","bisort","mebm25","ind49")

for (p in 1:length(port_rtn_list)){

  print("p")
  print(p)

  this_rtn = port_rtn_list[p]
  
  if (this_rtn == "mebm25"){
    data_y = mebm25
  } else if (this_rtn == "ind49"){
    data_y = ind49
  } else if (this_rtn == "bisort"){
    data_y = bisort
  } else if (this_rtn == "unisort"){
    data_y = unisort
  } else if (this_rtn == "pt1"){
    data_y = pt1
  } else if (this_rtn == "pt5"){
    data_y = pt5
  } else if (this_rtn == "pt10"){
    data_y = pt10
} else if (this_rtn == "pt15"){
    data_y = pt15
  } else if (this_rtn == "pt20"){
    data_y = pt20
  }

  port_r2 = rbind(port_r2, portR2.calc(data_y, data_pt_1, data_capm, "PT1", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); beta_regular_list = append(beta_regular_list, " "); min_n_factor = append(min_n_factor, " ")

  port_r2 = rbind(port_r2, portR2.calc(data_y, data_pt_5, data_capm, "PT5", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); beta_regular_list = append(beta_regular_list, " "); min_n_factor = append(min_n_factor, " ")
  
  port_r2 = rbind(port_r2, portR2.calc(data_y, data_pt_10, data_capm, "PT10", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); beta_regular_list = append(beta_regular_list, " "); min_n_factor = append(min_n_factor, " ")

  port_r2 = rbind(port_r2, portR2.calc(data_y, data_pt_15, data_capm, "PT15", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); beta_regular_list = append(beta_regular_list, " "); min_n_factor = append(min_n_factor, " ")

  port_r2 = rbind(port_r2, portR2.calc(data_y, data_pt_20, data_capm, "PT20", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); beta_regular_list = append(beta_regular_list, " "); min_n_factor = append(min_n_factor, " ")

  port_r2 = rbind(port_r2, portR2.calc(data_y, data_ff5, data_capm, "FF5", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); beta_regular_list = append(beta_regular_list, " "); min_n_factor = append(min_n_factor, " ")

  port_r2 = rbind(port_r2, portR2.calc(data_y, data_q5, data_capm, "Q5", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); beta_regular_list = append(beta_regular_list, " "); min_n_factor = append(min_n_factor, " ")

  port_r2 = rbind(port_r2, portR2.calc(data_y, data_rppca, data_capm, "RP5", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); beta_regular_list = append(beta_regular_list, " "); min_n_factor = append(min_n_factor, " ")

  port_r2 = rbind(port_r2, portR2.calc(data_y, data_ipca, data_capm, "IP5", this_rtn))
  g_list = append(g_list, " "); n_layer_list = append(n_layer_list, " "); n_factor_list = append(n_factor_list, " ")
  regularization_list = append(regularization_list, " "); beta_regular_list = append(beta_regular_list, " "); min_n_factor = append(min_n_factor, " ")

}

port_r2 = data.frame(port_r2)
colnames(port_r2) = c('port','model_list','instot','inspred','inscs')
port_r2 = port_r2[,c('port','model_list','instot','inspred','inscs')]
write.csv(data.frame(port_r2), paste0(output_path, 'Portfolio_R2_', case, '.csv'))

