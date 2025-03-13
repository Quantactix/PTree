library(MASS)

py_test = function(F1, F2){
  

  K = ncol(F1)
  N = ncol(F2)
  T = nrow(F2)
  X1 = t(F1)
  X2 = t(F2)
  ones = matrix(1,T,1)
  M_X1 = diag(T) - t(X1)%*%ginv((X1)%*%t(X1))%*%(X1)
  hat_alpha2 = X2%*%M_X1%*%ones%*%ginv(t(ones)%*%M_X1%*%ones)
  
  # 误差项
  full_X = cbind(ones,t(X1))
  M_F = diag(T) - full_X%*%ginv(t(full_X)%*%full_X)%*%t(full_X)
  
  error = X2%*%M_F
  error_Sigma = error%*%t(error)/(T - K -1)
  
  
  m = ginv(t(ones)%*%M_X1%*%ones)
  #hat_alpha0 = m%*%t(i_T)%*%M_z%*%F2
  var_alpha2 =  as.vector(m)*error_Sigma 
  T_stat2 = hat_alpha2^2/diag(var_alpha2)
  
  
  deg = T - K - 1
  
  rho2 = 0
  Rut = cor(t(error))
  pn = 0.1/(N-1)
  thetan = (qnorm(1-pn/2))^2
  rho2 = (sum((Rut[Rut^2*deg >thetan])^2)-N)/2
  rho2 = rho2*2/N/(N-1)
  HDA =  N^(-1/2)*(sum(T_stat2 - deg/(deg-2)))/((deg/(deg-2))*sqrt(2*(deg-1)/(deg-4) *(1+N*rho2)  )  )
  
  
  
  pvalue = 2*(1- pnorm(abs(HDA)))
  
  return(list(pvalue,HDA))
}