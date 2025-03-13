
source("py_test.R")

case = "yr_1981_2020_num_iter_9_boost_20"
folder = "P-Tree-a"

port_tree1      <- read.csv(paste0("./tmp/",folder,"/PT1.csv"),row.names = 1)
port_tree_top5  <- read.csv(paste0("./tmp/",folder,"/PT_top5.csv"),row.names = 1)
port_tree6_10   <- read.csv(paste0("./tmp/",folder,"/PT_top6_10.csv"),row.names = 1)
port_tree11_15  <- read.csv(paste0("./tmp/",folder,"/PT_top11_15.csv"),row.names = 1)
port_tree16_20  <- read.csv(paste0("./tmp/",folder,"/PT_top16_20.csv"),row.names = 1)
port_tree_top20 <- read.csv(paste0("./tmp/",folder,"/PT_top20.csv"),row.names = 1)


port_uni  <- read.csv(paste0("./tmp/",folder,"/Uni.csv"),row.names = 1)
port_bi   <- read.csv(paste0("./tmp/",folder,"/Bi.csv"),row.names = 1)
port_mebm <- read.csv(paste0("./tmp/",folder,"/mebm.csv"),row.names = 1)
port_ind  <- read.csv(paste0("./tmp/",folder,"/ind49.csv"),row.names = 1)

f <- read.csv(paste0("./tmp/",folder,"/FF.csv"),row.names = 1)

M <- matrix(rep(100,10*2), ncol = 2)

out <- py_test(f,port_tree1);out2 <- c(out[[1]], out[[2]])
M[1,] <- out2;

out <- py_test(f,port_tree_top5);out2 <- c(out[[1]], out[[2]])
M[2,] <- out2;

out <- py_test(f,port_tree6_10);out2 <- c(out[[1]], out[[2]])
M[3,] <- out2;

out <- py_test(f,port_tree11_15);out2 <- c(out[[1]], out[[2]])
M[4,] <- out2;

out <- py_test(f,port_tree16_20);out2 <- c(out[[1]], out[[2]])
M[5,] <- out2;

out <- py_test(f,port_tree_top20);out2 <- c(out[[1]], out[[2]])
M[6,] <- out2;

out <- py_test(f,port_uni);out2 <- c(out[[1]], out[[2]])
M[7,] <- out2;

out <- py_test(f,port_bi);out2 <- c(out[[1]], out[[2]])
M[8,] <- out2;

out <- py_test(f,port_mebm);out2 <- c(out[[1]], out[[2]])
M[9,] <- out2;

out <- py_test(f,port_ind);out2 <- c(out[[1]], out[[2]])
M[10,] <- out2;

df <- data.frame(M)
df2 <- df[,c('X1','X2')]
names(df2) <- c('p-value','t')
write.csv(df2, "tab3-a-py-test.csv")
