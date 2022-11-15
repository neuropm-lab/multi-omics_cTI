function W = local_scaling_affinityMatrix(Diff,Klocal)
% computes an affinity matrix for a given distance matrix
% Based on Duan et al., 2020. https://doi.org/10.1089/cmb.2019.0252

if nargin < 2
    Klocal = 5;
end

Diff = (Diff+Diff')/2;
Diff = Diff - diag(diag(Diff));
[T,INDEX] = sort(Diff,2);
[m,n] = size(Diff);
W = zeros(m,n);
TT = mean(sqrt(T(:,2:Klocal+1)),2)+eps; % TT = sqrt(T(:,Klocal+1))+eps; 
sigmaij = TT*TT';
sigmaij = sigmaij.*(sigmaij>eps)+eps;
W = normpdf(Diff,0, sigmaij);
W = (W + W')/2;
return
