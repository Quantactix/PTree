
# S=10000*2
# S=100
S <- 500

###### seed s ######
B = 10

# load all in-sample and out-of-sample random PTree portfolios

for (i in 1:S){
  print(i)

  for (j in 1:B){
    print(j)

    dir <- paste0("./trees/seed_",i,"/random_train_",j,".RData")
    print("a")
    if (file.exists(dir)){

      load(paste0("./trees/seed_",i,"/random_train_",j,".RData"))
      load(paste0("./trees/seed_",i,"/random_test_",j,".RData"))
      
      print(mean(pred1$ft)/sd(pred1$ft)*sqrt(12))
      
      eval(parse(text=paste0("ptree_train_",j,"<-fit1")))
      eval(parse(text=paste0("ptree_test_",j,"<-pred1")))
    }
  }
  rm(fit1,pred1)


  # make a big matrix of all the in-sample and out-of-sample portfolios

  p_in = ptree_train_1$portfolio
  for (j in 2:B){
    dir <- paste0("./trees/seed_",i,"/random_train_",j,".RData")
    print("b")
    if (file.exists(dir)){
      eval(parse(text=paste0("p_in = cbind(p_in,ptree_train_",j,"$portfolio)")))
    }
  }

  p_out = ptree_test_1$portfolio
  for (j in 2:B){
    dir <- paste0("./trees/seed_",i,"/random_train_",j,".RData")
    print("c")
    if (file.exists(dir)){
      eval(parse(text=paste0("p_out = cbind(p_out,ptree_test_",j,"$portfolio)")))
    }
  }

  write.csv(p_in, paste0("./trees/seed_",i,"/p_in.csv"))
  write.csv(p_out, paste0("./trees/seed_",i,"/p_out.csv"))

}