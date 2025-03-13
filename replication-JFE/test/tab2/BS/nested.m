%
%   nested.m
%   This MATLAB M-file computes the difference of squared
%   Sharpe ratios of two nested models as well 
%   as its p-values under conditional homosk. and conditional heterosk.
%
% Input:
% BigF: set of all factors
% m1: index for model 1
% m2: index for model 2
%
% Output:
% thetasqd: difference of sample squared Sharpe ratios of models 1 and 2
% pval1: p-value of testing H_0: theta_1^2=theta_2^2 under cond hom.
% pval2: p-value of testing H_0: theta_1^2=theta_2^2 under cond het.
%
function [thetasqd,pval1,pval2,F1,F2] = nested(BigF,m1,m2)
if length(m1)<length(m2)
   F1 = BigF(:,m1);
   F2 = BigF(:,setdiff(m2,m1));
else
   F1 = BigF(:,m2);
   F2 = BigF(:,setdiff(m1,m2));
end
index = any(isnan([F1 F2]),2);
F1(index,:) = [];
F2(index,:) = [];
F = [F1 F2];
[T,K1]=size(F1);
[T,K]=size(F);
mu1 = mean(F1)';
mu = mean(F)';
V11 = cov(F1,1);
V = cov(F,1);
W1 = (T-K1-2)./T*inv(V11);
W = (T-K-2)./T*inv(V);
theta2 = mu'*W*mu-K/T;
theta21 = mu1'*W1*mu1-K1/T;
thetasqd = theta2-theta21;
[F1,F2,pval1,pval2] = grs(F1,F2);
if length(m2)>length(m1)
   thetasqd = -thetasqd;
end
