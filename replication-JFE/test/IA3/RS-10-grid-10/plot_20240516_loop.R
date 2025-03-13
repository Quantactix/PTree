######################################################################
# function #
######################################################################

myplot<-function(df){
  max_df = max(df)
  
  i=1
  y = as.numeric(df[,i])
  x = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50, 100)
  
  plot(x,y,col=i,type = "l",
      lwd = 2, lty=1, ylim=c(0,max_df),
      xlab="c=P/T", ylab=" ")
  
  abline(v=1, lty=2)
  
  for (i in c(2:5)){
    y = as.numeric(df[,i])
    x = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50, 100)
    
    lines(x,y,col=i,type = "l",lwd = 2, lty=1)
  }
}

######################################################################
# main #
######################################################################

sum_mn = 0
sum_sm = 0
sum_sn = 0
sum_sr = 0
sum_hj = 0

NSim <- 20
# NSim <- 13

for (i in c(1:NSim)){
  
  load(paste0("./output/tab_1_20240516_complex_OOS_NSim_",i,".RData"))
  
  ######################################################################
  # Loop on each simulation and plot separate outcomes #
  ######################################################################

  # mean
  
  mn <- t(apply(tab,c(2,3),mean))
  mn=mn[,1:5]
  
  colnames(mn) <- c('z=1000','z=10','z=1','z=1e-1','z=1e-5')
  rownames(mn) <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50, 100)
  
  pdf(file = paste0("./output/mn_20240516_",i,".pdf"),
      width = 10,
      height = 10)
  myplot(mn)
  legend("right", 
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'), 
        col=1:5, pch=1, lwd = 3
        )
  title('Expected Return')
  dev.off()

  if (i==1){
        sum_mn = mn
  }else{
        sum_mn = sum_mn + mn
  }
  
  # std
  
  sn <- t(apply(tab,c(2,3),sd))
  sn=sn[,1:5]
  
  colnames(sn) <- c('z=1000','z=10','z=1','z=1e-1','z=1e-5')
  rownames(sn) <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50, 100)
  
  pdf(file = paste0("./output/sn_20240516_",i,".pdf"),
      width = 10,
      height = 10)
  myplot(sn)
  legend("right", 
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'), 
        col=1:5, pch=1, lwd = 3
        )
  title('Standard Deviation')
  dev.off()

  if (i==1){
        sum_sn = sn
  }else{
        sum_sn = sum_sn + sn
  }
  
  # 2nd moment
  
  sm <- t(apply(tab^2,c(2,3),mean))
  sm=sm[,1:5]
  
  colnames(sm) <- c('z=1000','z=10','z=1','z=1e-1','z=1e-5')
  rownames(sm) <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50, 100)
  
  pdf(file = paste0("./output/sm_20240516_",i,".pdf"),
      width = 10,
      height = 10)
  myplot(sm)
  legend("right", 
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'), 
        col=1:5, pch=1, lwd = 3
        )
  title('Second Moment')
  dev.off()

  if (i==1){
        sum_sm = sm
  }else{
        sum_sm = sum_sm + sm
  }
  
  # Sharpe ratio
  
  sr <- mn/sn*sqrt(12)
  sr = sr[,1:5]
  colnames(sr) <- c('z=1000','z=10','z=1','z=1e-1','z=1e-5')
  rownames(sr) <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50, 100)
  
  pdf(file = paste0("./output/sr_20240516_",i,".pdf"),
      width = 10,
      height = 10)
  myplot(sr)
  legend("right", 
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'), 
        col=1:5, pch=1, lwd = 3
        )
  title('Sharpe Ratio')
  dev.off()

  if (i==1){
        sum_sr = sr
  }else{
        sum_sr = sum_sr + sr
  }

  # HJD 

  one_minus_tab = 1-tab
  
  hj <- t(apply(one_minus_tab^2,c(2,3),mean))
  hj=hj[,1:5]
  
  colnames(hj) <- c('z=1000','z=10','z=1','z=1e-1','z=1e-5')
  rownames(hj) <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50, 100)
  
  pdf(file = paste0("./output/hj_20240516_",i,".pdf"),
      width = 10,
      height = 10)
  myplot(hj)
  legend("right", 
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'), 
        col=1:5, pch=1, lwd = 3
        )
  title('Pricing Error')
  dev.off()

  if (i==1){
        sum_hj = hj
  }else{
        sum_hj = sum_hj + hj
  }
  
}




######################################################################
# Plot AVG outcomes #
######################################################################

mn = sum_mn / NSim
sm = sum_sm / NSim
sn = sum_sn / NSim
sr = sum_sr / NSim
hj = sum_hj / NSim

pdf(file = './output/mn_20240516_avg20.pdf',
    width = 10,
    height = 10)

myplot(mn)
legend("right",
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'),
        col=1:5, pch=1, lwd = 3
        )
title('Expected Return')
dev.off()


pdf(file = './output/sn_20240516_avg20.pdf',
    width = 10,
    height = 10)

myplot(sn)
legend("right",
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'),
        col=1:5, pch=1, lwd = 3
        )
title('Standard Deviation')
dev.off()


pdf(file = './output/sm_20240516_avg20.pdf',
    width = 10,
    height = 10)

myplot(sm)
legend("right",
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'),
        col=1:5, pch=1, lwd = 3
        )
title('Second Moment')
dev.off()


pdf(file = './output/sr_20240516_avg20.pdf',
    width = 10,
    height = 10)

myplot(sr)
legend("right",
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'),
        col=1:5, pch=1, lwd = 3
        )
title('Sharpe Ratio')
dev.off()


pdf(file = './output/hj_20240516_avg20.pdf',
    width = 10,
    height = 10)

myplot(hj)
legend("right",
        legend = c('z=1000','z=10','z=1','z=1e-1','z=1e-5'),
        col=1:5, pch=1, lwd = 3
        )
title('Pricing Error')
dev.off()

write.csv(mn,"./output/mn_20240516_avg20.csv")
write.csv(sm,"./output/sm_20240516_avg20.csv")
write.csv(sn,"./output/sn_20240516_avg20.csv")
write.csv(sr,"./output/sr_20240516_avg20.csv") 
write.csv(hj,"./output/hj_20240516_avg20.csv") 
