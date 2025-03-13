
see <- 100000 

for (i in 1:see){
  print("see")
  print(i)

  if (!file.exists(paste0("./trees/seed_",i,"/random_train.RData"))){
    next
  }

  load(paste0("./trees/seed_",i,"/random_train.RData"))
  load(paste0("./trees/seed_",i,"/random_test.RData"))

  print(mean(fit1$ft)/sd(fit1$ft)*sqrt(12))
  print(mean(pred1$ft)/sd(pred1$ft)*sqrt(12))
  
  eval(parse(text=paste0("ptree_train_seed_",i,"<-fit1$portfolio")))
  eval(parse(text=paste0("ptree_test_seed_" ,i,"<-pred1$portfolio")))
}

rm(fit1,pred1)

# make a big matrix of all the in-sample and out-of-sample portfolios

p_in = ptree_train_seed_1
for (i in 2:see){
  print("p_in = ptree_train_seed_1")
  print(i)

  if (!file.exists(paste0("./trees/seed_",i,"/random_train.RData"))){
    next
  }

  eval(parse(text=paste0("p_in = cbind(p_in,ptree_train_seed_",i,")")))
}

p_out = ptree_test_seed_1
for (i in 2:see){
  print("p_out = ptree_test_seed_1")
  print(i)

  if (!file.exists(paste0("./trees/seed_",i,"/random_train.RData"))){
    next
  }

  eval(parse(text=paste0("p_out = cbind(p_out,ptree_test_seed_",i,")")))
}

# solve SDF weight

T <- dim(p_in)[1]
P <- dim(p_in)[2]

print("T")
print(T)

print("P")
print(P)

write.csv(p_in, paste0("./output/rpf_in_1_",see,".csv"))
write.csv(p_out, paste0("./output/rpf_out_1_",see,".csv"))

save(p_in, p_out, file = paste0("./output/rpf_1_",see,".RData"))
