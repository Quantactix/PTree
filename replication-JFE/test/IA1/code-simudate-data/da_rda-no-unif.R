
BB = 11 - 1

library(arrow)

for (i in c(2, 0.5, 1)){
# for (i in c(1)){
    kappa = i
    print(kappa)
    print("#####")
    for (seq in c(1:BB)){
        print("seq")
        print(seq)
        data_src = paste0("../data/simu_kappa_",kappa,"_seed_",seq,".feather")
        data_out = paste0("../data/simu_kappa_",kappa,"_seed_",seq,".RData")
        data = read_feather(data_src)
        save(data,file=data_out)
    }
}
