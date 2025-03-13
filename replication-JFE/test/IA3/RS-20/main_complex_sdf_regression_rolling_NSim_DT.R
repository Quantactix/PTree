################################################################################################
# Divide by T #####################################################################################
################################################################################################

# Multiple Simulation.
#  
# This file combines the ridge svd funx for P < T and get beta funx for P > T.
# Reference
# "The Virtue of Complexity in Return Prediction BRYAN KELLY, SEMYON MALAMUD, and KANGYING ZHOU*"
# 
# Estimation with 240-month rolling-window data
# Output one-month-ahead OOS SDF return.
################################################################################################
# Function #####################################################################################
################################################################################################

ridgesvd <- function(Y, X, lambda) {
    # Check for missing data
    if (any(is.na(X)) || any(is.na(Y))) {
    stop('missing data')
    }

    L <- length(lambda)
    #   svdX <- svd(X)
    svdX <- svd(X, nu=nrow(X), nv=ncol(X))
    U <- svdX$u
    D <- svdX$d
    V <- svdX$v
    T <- nrow(X)
    P <- ncol(X)

    # Initialization of B
    B <- matrix(NA, nrow = P, ncol = L)

    # Conditional append for diagonal matrix construction
    if (T >= P) {
    compl <- matrix(0, nrow = P, ncol = T - P)
    } else {
    compl <- matrix(0, nrow = P - T, ncol = T)
    }

  # Computation of B for each lambda
    for (l in 1:L) {
        
        diag_part <- diag(D / (D^2 + lambda[l]))

        if (T >= P) {   # which should not be activated
            matrix_lambda <- cbind(diag_part, compl)
            B[, l] <- V %*% matrix_lambda %*% t(U) %*% Y
        } else {        # which should not be activated
            matrix_lambda <- rbind(diag_part, compl)
            B[, l] <- V %*% matrix_lambda %*% t(U) %*% Y
        }

    }
  
    return(B)
}

get_beta <- function(Y, X, lambda_list) {
  # Check for missing data
  if (any(is.na(c(X))) || any(is.na(Y))) {
    stop('missing data')
  }
  
  L_ <- length(lambda_list)
  T_ <- nrow(X)
  P_ <- ncol(X)
  
  # Determine the scaling and decomposition based on the dimension of X
  if (P_ > T_) {
    a_matrix <- (X %*% t(X)) / T_  # T_ x T_
  } else {
    a_matrix <- (t(X) %*% X) / T_  # P_ x P_
  }
  
  svd_a <- svd(a_matrix)
  U_a <- svd_a$u
  D_a <- svd_a$d
  V_a <- svd_a$v
  
  # Scale eigenvalues for whitening transformation
  scale_eigval <- diag((D_a * T_)^(-1/2))
  
  # Whitened X' matrix
  W <- t(X) %*% U_a %*% scale_eigval
  
  # Eigenvalues of a_matrix
  a_matrix_eigval <- D_a
  
  # Compute signal times return
  signal_times_return <- t(X) %*% Y / T_  # M x 1
  signal_times_return_times_v <- t(W) %*% signal_times_return  # T x 1
  
  # Initialize B
  B <- matrix(NA, nrow = P_, ncol = L_)
  
  # Compute B for each lambda
  for (l in 1:L_) {
    B[, l] <- W %*% diag(1 / (a_matrix_eigval + lambda_list[l])) %*% signal_times_return_times_v
  }
  
  return(B)
}

calc.sdf.at.cpt<-function(p_in, p_out){
    
  T <- dim(p_in)[1]
  P <- dim(p_in)[2]

  if (P > T) {
    # High Complexity, get beta funx
    w_p3 <- get_beta(matrix(rep(1,T),nrow=T), p_in, 1e3/T)
    w_p1 <- get_beta(matrix(rep(1,T),nrow=T), p_in, 1e1/T)
    w_p0 <- get_beta(matrix(rep(1,T),nrow=T), p_in, 1e0/T)
    w_m1 <- get_beta(matrix(rep(1,T),nrow=T), p_in, 1e-1/T)
    w_m5 <- get_beta(matrix(rep(1,T),nrow=T), p_in, 1e-5/T)
  }else{
    # Low Complexity, ridge svd funx
    w_p3 <- ridgesvd(matrix(rep(1,T),nrow=T), p_in, 1e3)
    w_p1 <- ridgesvd(matrix(rep(1,T),nrow=T), p_in, 1e1)
    w_p0 <- ridgesvd(matrix(rep(1,T),nrow=T), p_in, 1e0)
    w_m1 <- ridgesvd(matrix(rep(1,T),nrow=T), p_in, 1e-1)
    w_m5 <- ridgesvd(matrix(rep(1,T),nrow=T), p_in, 1e-5)
  }

  # make SDF

  sdf_in_p3 = as.matrix(p_in) %*% as.matrix(w_p3)
  sdf_in_p1 = as.matrix(p_in) %*% as.matrix(w_p1)
  sdf_in_p0 = as.matrix(p_in) %*% as.matrix(w_p0)
  sdf_in_m1 = as.matrix(p_in) %*% as.matrix(w_m1)
  sdf_in_m5 = as.matrix(p_in) %*% as.matrix(w_m5)
 
  sdf_out_p3 = as.matrix(p_out) %*% as.matrix(w_p3)
  sdf_out_p1 = as.matrix(p_out) %*% as.matrix(w_p1)
  sdf_out_p0 = as.matrix(p_out) %*% as.matrix(w_p0)
  sdf_out_m1 = as.matrix(p_out) %*% as.matrix(w_m1)
  sdf_out_m5 = as.matrix(p_out) %*% as.matrix(w_m5)
  
  # report oos SDF performance

  sr<-function(sdf){
    mn <- mean(sdf)
    sn <- sd(sdf)
    sr <- mean(sdf)/sd(sdf)*sqrt(12)
    return(c(mn,sn,sr))
    # return(sr)
  }

  sr(sdf_out_p3)
  sr(sdf_out_p1)
  sr(sdf_out_p0)
  sr(sdf_out_m1)
  sr(sdf_out_m5)
  
  # df<-data.frame(
  #   cbind(  sr(sdf_out_p3),
  #           sr(sdf_out_p1),
  #           sr(sdf_out_p0),
  #           sr(sdf_out_m1),
  #           sr(sdf_out_m5)
  #           ))

  df <- data.frame(
        cbind(  sdf_out_p3,
                sdf_out_p1,
                sdf_out_p0,
                sdf_out_m1,
                sdf_out_m5
        ))

  return(df)
}

################################################################################################
# Main #########################################################################################
################################################################################################

NSim <- 20       # each simulation goes to c=P/T=100.
Sim  <- 1        # the sequence of simulation. 1 to 20.
NTree<- 240     # which makes 240 trees and 2400 random leaves, thus c=100.
date <- 20240516

load("./output/p_1_340.RData")

p_all_full <- rbind(p_in,p_out)
big_T <- 240

# Allocate the data columns for one simulation.

for (Sim in c(1:20)){

  print("One Sim")
  print(Sim)

  p_all <- p_all_full[,((Sim-1)*big_T*10+1):(Sim*big_T*10)]
  # print(dim(p_all))

  # complexity $c=P/T$
  
  cpt_list <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  # cpt_list <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 10, 20, 50, 100)
  # cpt_list <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
  #             0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
  #             1,
  #             1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 
  #             1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
  #             2,
  #             3,4,5,6,7,8,9,10)
  #             ,15,20,50)
  
  l        <- length(cpt_list)
  print(cpt_list)

  n_penalty<- 5

  tab <- array(rep(-100,big_T*n_penalty*l), dim = c(big_T, n_penalty, l)) 
  tab

  # rwt (rolling window t)
  # oos: 241 to 480.

  # rwt_list <- c(241:250)
  rwt_list <- c(241:480)

  t=1
  for (rwt in rwt_list){
    print(rwt)
    # p_in <- p_all[1:(rwt-1),]
    # p_out<- p_all[rwt,]

    i=1
    for (cpt in cpt_list){  
      print("cpt")
      print(cpt)
      print(Sys.time())
      
      # cpt = 10
      this_P <- cpt * big_T
      # print("this_P")
      # print(this_P)

      # this_P <- 239
      p_in <- p_all[(rwt-big_T):(rwt-1),1:this_P]
      p_out<- p_all[rwt,1:this_P]

      df <- calc.sdf.at.cpt(as.matrix(p_in),as.matrix(p_out))
      # print(as.matrix(df))
      # print(dim(df))
      tab[t,,i] <- as.matrix(df)
      # print(tab)

      i=i+1
      # print("i")
      # print(i)
    }
    # print("Final tab")
    # print(tab)
    save(tab, file = paste0("./output/tab_1_",date,"_complex_NSim_",Sim,"_rwt_",rwt,"_.RData"))
    
    t=t+1
    # print("t")
    # print(t)
  }

  save(tab, file = paste0("./output/tab_1_",date,"_complex_OOS_NSim_",Sim,".RData"))
  print("\n Finish a Sim \n")
  print(Sim)
  print("\n Finish a Sim \n")

}

print("\n Good Job! \n -- Xin He. \n 20240516. \n")

############################################

# load("tab_1_20240416_complex_OOS_.RData")
# tab1 = tab[1:100,,]

# sr = apply(tab,c(2,3),mean)/apply(tab,c(2,3),sd)
# print(sr * sqrt(12))
