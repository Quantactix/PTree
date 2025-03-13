%
%	GRS.M
%	This Matlab m-file computes the Gibbons-Ross-Shanken F-test of the
%	efficiency of K-benchmark portfolios r1 with respect to N test portfolios r2.
%	Both conditional homoskedastic and conditional heteroskedastic versions are
%	computed.
%
function [F1,F2,pval1,pval2] = grs(r1,r2)
[T,K] = size(r1);
[T,N] = size(r2);
W1 = inv(cov(r1,1));
SR1 = 1+mean(r1)*W1*mean(r1)';	% 1+SR^2 of tangent portfolio of r1
r = [r1 r2];
W = inv(cov(r,1));
SR = 1+mean(r)*W*mean(r)';		% 1+SR^2 of tangeet portfolio of r
F1 = ((T-K-N)/N)*(SR/SR1-1);
pval1 = 1-fcdf(F1,N,T-K-N);
X = [ones(T,1) r1];
B = X\r2;
E = r2-X*B;
a = B(1,:)';
e1 = [1; zeros(K,1)];
U = X*inv(X'*X)*e1;
Ue = E.*(U*ones(1,N));
Ga = (Ue'*Ue);
F2 = ((T-K-N)/(N*T))*(a'*inv(Ga)*a);
pval2 = 1-fcdf(F2,N,T-K-N);
