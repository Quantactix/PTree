clear;
clc;

folder = "P-Tree-f";
case1 = 'Gyr_2001_2020__1981_2000_num_iter_9_boost_20';
filename = "../../"+folder+"/"+case1+"_bf_train.csv";

M = readmatrix(filename);
M = M(2:end,2:end); % remove index column

list_thetasqd = zeros(21,1);
list_p1= zeros(21,1);
list_p2= zeros(21,1);
list_f1= zeros(21,1);
list_f2= zeros(21,1);

for i = 2:21   % % Sepcification of remove index column
    F = M(1:end,1:i);
    idx2 = 1:(i-1);
    idx1 = 1:(i);
    [thetasqd,pval1,pval2,F1,F2] = nested(F,idx1,idx2);
    
    list_thetasqd(i) = thetasqd;
    list_p1(i) = pval1;
    list_p2(i) = pval2;
    list_f1(i) = F1;
    list_f2(i) = F2;
end

out = [list_thetasqd, list_p1, list_p2, list_f1, list_f2];
disp(out);
writematrix(out,'BS-test-'+folder+'.csv')
