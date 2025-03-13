
# load all in-sample and out-of-sample random PTree portfolios from 10 seeds

# see <- 7000
# see <- 10000*2
see <- 340 

for (i in 1:see){
  print("see")
  print(i)
  
  dir <- paste0("./trees/seed_",i,"/p_in.csv")
  if (file.exists(dir)){
    p_in_see  <- read.csv(paste0("./trees/seed_",i,"/p_in.csv"))
    p_out_see <- read.csv(paste0("./trees/seed_",i,"/p_out.csv"))    
  
    p_in_see  <- p_in_see[,c(-1)]
    p_out_see <- p_out_see[,c(-1)]
    
    eval(parse(text=paste0("ptree_train_seed_",i,"<-p_in_see")))
    eval(parse(text=paste0("ptree_test_seed_" ,i,"<-p_out_see")))
  }

}

rm(p_in_see,p_out_see)

# make a big matrix of all the in-sample and out-of-sample portfolios

p_in = ptree_train_seed_1
for (i in 2:see){
  print(i)
  if (file.exists(dir)){
    eval(parse(text=paste0("p_in = cbind(p_in,ptree_train_seed_",i,")")))
  }
}

p_out = ptree_test_seed_1
for (i in 2:see){
  print(i)
  if (file.exists(dir)){
    eval(parse(text=paste0("p_out = cbind(p_out,ptree_test_seed_",i,")")))
  }
}

# solve SDF weight

T <- dim(p_in)[1]
P <- dim(p_in)[2]

print("T")
print(T)

print("P")
print(P)

write.csv(p_in, paste0("./trees/p_in_1_",see,".csv"))
write.csv(p_out, paste0("./trees/p_out_1_",see,".csv"))

save(p_in, p_out, file = paste0("./output/p_1_",see,".RData"))
