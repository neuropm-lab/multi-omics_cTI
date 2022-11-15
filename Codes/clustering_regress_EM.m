function [Clust,Probs,R2_clusters,P_clusters,BIC,R2_nonspecific] = clustering_regress_EM(Y,data,C0,max_repetition,min_points_regression)
% Yasser Iturria Medina,
% Montreal, Feb. 25th, 2021.

% Copyright (c) ------------------------------------------------------------------% 
% Dr. Yasser Iturria-Medina, the NeuroInformatics for Personalized Medicine 
% lab (the NeuroPM-lab), McGill University. 2021.
% Maintainer: yasser.iturriamedina@mcgill.ca, iturria.medina@gmail.com.

% THE SOFTWARE IS DISTRIBUTED "AS IS" UNDER THIS LICENSE SOLELY FOR NON-COMMERCIAL 
% USE IN THE HOPE THAT IT WILL BE USEFUL, BUT IN ORDER THAT THE UNIVERSITY AS A 
% CHARITABLE FOUNDATION PROTECTS ITS ASSETS FOR THE BENEFIT OF ITS EDUCATIONAL 
% AND RESEARCH PURPOSES, THE NEUROPM-LAB MAKES CLEAR THAT NO CONDITION IS MADE OR 
% TO BE IMPLIED, NOR IS ANY WARRANTY GIVEN OR TO BE IMPLIED, AS TO THE ACCURACY OF 
% THE SOFTWARE, OR THAT IT WILL BE SUITABLE FOR ANY PARTICULAR PURPOSE OR FOR USE 
% UNDER ANY SPECIFIC CONDITIONS. FURTHERMORE, THE NEUROPM-LAB DISCLAIMS ALL RESPONSIBILITY 
% FOR THE USE WHICH IS MADE OF THE SOFTWARE. IT FURTHER DISCLAIMS ANY LIABILITY 
% FOR THE OUTCOMES ARISING FROM USING THE SOFTWARE.
%------------------------------------------------------------------------------------%

rng('default'); % For reproducibility
Y      = boxcox(Y - min(Y) + eps);
[N,Nf] = size(data);

if nargin < 3 || isempty(C0), C0 = spectralcluster(data,6); end
if nargin < 4, max_repetition = 500; end
if nargin < 5, min_points_regression = 20; end

C = C0; Clust = C0; sigma_Y = std(Y); nan_data = isnan(data); repetition = 0; 
while 1
    repetition = repetition + 1;
    unique_C = unique(C); Nclust = length(unique_C); P = zeros(N,Nclust); clear Probs;
    for i = 1:N % To be equivalent to leave-one-out.
        i_nonan = find(~nan_data(i,:));
        for j = unique_C(:)'
            train = setdiff(find(C==j & sum(nan_data(:,i_nonan),2)==0), i);
            if length(train) >= min_points_regression
                B = [ones(length(train),1) data(train,i_nonan)]\Y(train); % B = regress(Y(train),data(train,i_nan));
                Yhat(i,j==unique_C) = [1 data(i,i_nonan)]*B;
                P(i,j==unique_C)    = (1/(sigma_Y*sqrt(pi)))*exp(-(Yhat(i,j==unique_C) - Y(i))^2/(2*sigma_Y^2));                
            else
                P(i,j==unique_C) = 0;
            end
        end
        [Probs(i,:),j] = sort(P(i,:),'descend');
        if max_repetition > 1, % updating/improving subtyping based on probabilities.
            Clust(i,1:length(j)) = j;
        end
    end
    clear C_Prob; for j = unique_C(:)', C_Prob(j==unique_C) = mean(Probs(Clust(:,1)==j,1)); end
    CONV1 = sum(C == Clust(:,1))/N; CONV2 = sum(Probs(:,1))/N;
    disp(['Subtyping iteration -> ' num2str(repetition) ', clusters averg. Prob. -> ' num2str(C_Prob) ', stability -> ' num2str(CONV1)]);
    
    if CONV1 > 0.99 || repetition > max_repetition,
        break;
    else
        C = Clust(:,1); CONV2_ant = CONV2;
    end
end
% Calculating cluster-specific prediction R2.
Clust = Clust(:,1); clear Yhat Yhat_nonspecific
for i = 1:N % to be equivalent to leave-one-out.
    i_nonan = find(~nan_data(i,:));
    train = setdiff(find(Clust==Clust(i) & sum(nan_data(:,i_nonan),2)==0), i);
    if length(train) >= min_points_regression
        B = [ones(length(train),1) data(train,i_nonan)]\Y(train); % B = regress(Y(train),data(train,i_nan));
        Yhat(i,1) = [1 data(i,i_nonan)]*B;
    else
        Yhat(i,1) = NaN;
    end
    % Taking instead all subjects as predictors
    train = setdiff(find(sum(nan_data(:,i_nonan),2)==0), i);
    B = [ones(length(train),1) data(train,i_nonan)]\Y(train); % B = regress(Y(train),data(train,i_nan));
    Yhat_nonspecific(i,1) = [1 data(i,i_nonan)]*B;
end
for j = unique(Clust(:))', 
    [R,P_clusters(j)] = corr(Y(Clust==j),Yhat(Clust==j)); 
    R2_clusters(j) = 100*(R^2); residual(Clust==j) = (Y(Clust==j) - Yhat(Clust==j)); 
end
[R_nonspecific,P_nonspecific] = corr(Y,Yhat_nonspecific); R2_nonspecific = 100*(R_nonspecific^2);
k = (1+size(data,2))*length(unique(Clust(:))); % residual(isnan(residual)) = [];
L = (1/(sigma_Y*sqrt(pi)))*exp(-(residual*residual')/(2*sigma_Y^2));
BIC = k*log(N) - 2*log(L); % https://en.wikipedia.org/wiki/Bayesian_information_criterion
return;