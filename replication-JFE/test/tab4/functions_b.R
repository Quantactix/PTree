##### Judge Significance #####
judge.star <- function(value){
  if (value > 0.99){
    output = "***"
  } else if (value > 0.95){
    output = "**"
  } else if (value > 0.90){
    output = "*"
  } else{
    output = ""
  }
  return(output)
}



##### Define functions to calculate Total R2 and Predictive R2 #####
R2.calc = function(y, ypred, ypred_capm){
  return(100*(1 - sum((y - ypred)^2) / sum((y - ypred_capm)^2)))
}

##### Define functions to calculate Cross-Sectional R2 #####
CSR2.calc = function(stocks_index, y, haty, haty_capm){
  df <- data.frame(cbind(stocks_index, y, haty, haty_capm))
  colnames(df) = c('stocks_index','y','haty','haty_capm')
  df$e=as.numeric(df$y)-as.numeric(df$haty)
  df$ecapm = as.numeric(df$y) - as.numeric(df$haty_capm)
  numer <- aggregate(x=df$e, by=list(stocks_index), FUN=mean)$x
  denom <- aggregate(x=df$ecapm, by=list(stocks_index), FUN=mean)$x
  return((1-mean(numer**2)/mean(denom**2))*100)
}



##### Calculate Portfolio R2 #####
portR2.calc <- function(df_y, df_x, mat_capm, model, rtn_name){
  y_ins = df_y[1:240,]
  x_ins = df_x[1:240,]
  capm_ins = mat_capm[1:240,]
  
  y_total = c(); y_total_test = c()
  haty_total = c(); haty_total_test = c()
  haty_capm_total = c(); haty_capm_total_test = c()
  haty_total_pred = c(); haty_total_pred_test = c()
  haty_capm_total_pred = c(); haty_capm_total_pred_test = c()
  stocks_newlist = c(); stocks_newlist_test = c()

  y_bar_total = c(); y_bar_total_test = c()
  beta_model = c(); beta_capm = c()
  
  for (i in 1:dim(df_y)[2]){
    stocks_newlist = append(stocks_newlist, rep(colnames(df_y)[i], dim(y_ins)[1]))
    
    # In-Sample Total R2 and Cross-Sectional R2
    y <- y_ins[,i]
    x <- as.matrix(x_ins)
    avg_x_train <- apply(x, 2, mean)
    
    ins_reg <- summary(lm(y ~ x))#; print(ins_reg)
    b_model <- ins_reg$coefficients[,1]
    # haty <- x%*%as.matrix(b_model[2:length(b_model)]) # 20230915
    haty <- x%*%as.matrix(b_model[2:length(b_model)]) + b_model[1]
    y_total <- append(y_total, as.numeric(y))
    haty_total <- append(haty_total, as.numeric(haty)); rm("haty")
    
    y_bar_total <- append(y_bar_total, mean(y))
    beta_model <- rbind(beta_model, b_model[2:length(b_model)])
    
    ins_capm_reg <- summary(lm(y ~ as.matrix(capm_ins)))#; print(ins_capm_reg)
    b_capm <- ins_capm_reg$coefficients[,1]
    # haty_capm <- as.matrix(capm_ins)%*%as.matrix(b_capm[2:length(b_capm)]) # 20230915
    haty_capm <- as.matrix(capm_ins)%*%as.matrix(b_capm[2:length(b_capm)]) + b_capm[1]
    haty_capm_total <- append(haty_capm_total, as.numeric(haty_capm)); rm("haty_capm")
    beta_capm <- append(beta_capm, b_capm[2])
    rm("x")
    
    # In-Sample Predictive R2

    # print("dim(df_x)[1]")
    # print(dim(df_x)[1])

    x <- matrix(rep(c(1,avg_x_train), dim(df_x)[1]), nrow = dim(df_x)[1], ncol = 1+length(avg_x_train), byrow = TRUE)
    haty <- x%*%as.matrix(b_model)
    haty_total_pred <- append(haty_total_pred, as.numeric(haty)); rm("haty")
    
    haty_capm <- as.matrix(rep(mean(capm_ins), dim(df_x)[1]))%*%as.matrix(b_capm[2:length(b_capm)])
    haty_capm_total_pred <- append(haty_capm_total_pred, as.numeric(haty_capm)); rm("haty_capm")
    rm("x")
  }

  # print("beta_model")
  # print(beta_model)
  
  # Calculate risk premia estimates
  # reg_lambda = summary(lm(y_bar_total ~ beta_model)) # 20230915
  reg_lambda = summary(lm(y_bar_total ~ beta_model - 1)) # no intercept

  # print('reg_lambda')
  # print(reg_lambda)

  # print("reg_lambda$coefficients")
  # print(reg_lambda$coefficients)

  # lambda_val = reg_lambda$coefficients[,1]; lambda_val = lambda_val[2:length(lambda_val)]  # 20230915
  lambda_val = reg_lambda$coefficients[,1]

  # print("lambda_val")
  # print(lambda_val)

  haty_bar_total = as.numeric(as.matrix(beta_model)%*%as.matrix(lambda_val))
  
  # reg_lambda_capm = summary(lm(y_bar_total ~ beta_capm)) # 20230915
  reg_lambda_capm = summary(lm(y_bar_total ~ beta_capm - 1)) # no intercept
  # lambda_val_capm = reg_lambda_capm$coefficients[,1]; lambda_val_capm = lambda_val_capm[2] # 20230915
  lambda_val_capm = reg_lambda_capm$coefficients[,1]; 
  haty_bar_capm_total = as.numeric(lambda_val_capm*beta_capm)
  
  haty_bar_total_test = as.numeric(as.matrix(beta_model)%*%as.matrix(lambda_val))
  haty_bar_capm_total_test = as.numeric(lambda_val_capm*beta_capm)
  
  # Start to output R2 results and save
  # print(paste0("Start to calculate ", rtn_name, " Portfolio R2."))
  # ins_TotR2 = R2.calc(y_total, haty_total, haty_capm_total)
  ins_TotR2 = R2.calc(y_total, haty_total, 0)
  print(paste0(model, " In-Sample Total R2 %: "))
  print(ins_TotR2)

  # ins_PredR2 = R2.calc(y_total, haty_total_pred, haty_capm_total_pred)
  ins_PredR2 = R2.calc(y_total, haty_total_pred, 0)
  print(paste0(model, " In-Sample Predictive R2 %: "))
  print(ins_PredR2)

  # ins_CSR2_other = R2.calc(y_bar_total, haty_bar_total, haty_bar_capm_total)
  ins_CSR2_other = R2.calc(y_bar_total, haty_bar_total, 0)
  print(paste0(model, " In-Sample Cross-Sectional R2 % (another way): "))
  print(ins_CSR2_other)

  # plot
  print("\n Plot of haty_bar_total and y_bar_total \n")
  
  pdf(file = paste0("./figure/",rtn_name,"-",model,".pdf"),
      width = 5,
      height = 5)
  par(mar=c(2,2,0.5,0.5)) # bottom, left, top, right
  plot(haty_bar_total*100, y_bar_total*100, 
      # main=paste0(model,"-",rtn_name),
      # main=model,
      xlab = "", ylab = "",
      xlim=c(-0.4,1.3), ylim=c(-0.4,1.3)
      ) 
  abline(0,1)
  dev.off()
      

  return(c(rtn_name, model, ins_TotR2, ins_PredR2, ins_CSR2_other))
}
