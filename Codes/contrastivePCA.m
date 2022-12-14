function [cPCs,gap_values,alphas_f,no_dims,contrasted_data,Vmedoid,Dmedoid] = contrastivePCA(X,indices_background,indices_target,d_max,classes_for_colours,alphas)
% Based on Abid et al., 2018, Nature Comms., 9:2134.

% Example data:
% n = 200; k = 10;
% X = [[mvnrnd([0 0 0],[1 1 1],n) 0.1*randn(n,50)]; ...
%     [mvnrnd([2 2 2],ones(3,3),k*n) 0.1*randn(k*n,50)]; ...
%     [mvnrnd([4 4 4],ones(3,3),n) 0.1*randn(n,50)]];
% indices_background  = [1:n]';
% indices_target      = [n+1:size(X,1)]';
% classes_for_colours = [ones(n,1); 2*ones(k*n,1); 3*ones(n,1)];

% Copyright (c) ------------------------------------------------------------------% 
% Dr. Yasser Iturria-Medina, the NeuroInformatics for Personalized Medicine 
% lab (the NeuroPM-lab), McGill University. 2020.
% Maintainer: yasser.iturriamedina@mcgill.ca, iturria.medina@gmail.com

% The NeuroPM-box remains the property of the NeuroPM-lab. No part of the Software 
% may be reproduced, modified, transmitted or transferred in any form or by any means, 
% electronic or mechanical, without the express permission of the NeuroPM-lab?s Director 
% (yasser.iturriamedina@mcgill.ca). You may be held legally responsible for any 
% copyright infringement that is caused or encouraged by your failure to abide by 
% these terms and conditions.

% You are not permitted under this License to use this Software commercially. Use for 
% which any financial return is received shall be defined as commercial use, and 
% includes (1) integration of all or part of the source code or the Software into 
% a product for sale or license by or on behalf of Licensee to third parties or 
% (2) use of the Software or any derivative of it for research with the final aim 
% of developing software products for sale or license to a third party or (3) use 
% of the Software or any derivative of it for research with the final aim of 
% developing non-software products for sale or license to a third party, or 
% (4) use of the Software to provide any service to an external organization 
% for which payment is received. 

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

rng('default');  % For reproducibility
X = zscore(X); % Transforming and centering data
if nargin < 4
    d_max = min([size(X,2) 10]);
end
if ~exist('classes_for_colours') || isempty(classes_for_colours)
    classes_for_colours = zeros(size(X,1),1);
    classes_for_colours(indices_background,1) = 1;
    classes_for_colours(indices_target,1)     = 2;
else
    classes_for_colours = classes_for_colours(:);
end
if nargin < 6 || isempty(alphas)
    alphas = logspace(-2,2,100);
end
n_alphas = length(alphas);

% Transforming non-normal features into a normal shape (didn't change our results, but may be help with some data).
for d = 1:size(X,2)
    [~,lambda] = boxcox(X([indices_background; indices_target],d)-min(X(:,d))+eps);
    if lambda > 0,
        X(:,d) = ((X(:,d)-min(X(:,d))+eps).^lambda - 1)/lambda;
    elseif lambda == 0,
        X(:,d) = log(X(:,d)-min(X(:,d))+eps);
    end
end
mean_data = mean(X); std_data = std(X);
X = (X - repmat(mean(X([indices_background; indices_target],:)),[size(X,1) 1]))./ ...
    repmat(std(X([indices_background; indices_target],:)) + eps,[size(X,1) 1]); % Standardizing data after box-cox

% % Detecting and removing significant outliers in background and target
% A_background = isoutlier(X(indices_background,:)); B_background = find(isoutlier(sum(A_background,2))); indices_background(B_background) = [];
% A_target = isoutlier(X(indices_target,:));         B_target     = find(isoutlier(sum(A_target,2)));     indices_target(B_target) = [];

X_background = X(indices_background,:); % X_background = filloutliers(X_background,'spline');
X_target     = X(indices_target,:);     % X_target     = filloutliers(X_target,'spline');

% Covariance matrices
Cb = cov(X_background); % cov(zscore(X_background));
Ct = cov(X_target);     % cov(zscore(X_target));

% cPCA with autoselection of alpha_f:
for alpha_i = 1:n_alphas
    if alpha_i == 1,
        beta = regress(Ct(:),Cb(:));
        C = Ct - beta*Cb; alphas(alpha_i) = beta;
    else
        C = (Ct - alphas(alpha_i)*Cb);
    end
    C(isnan(C)) = 0;
    C(isinf(C)) = 0;
    rng('default');  % For reproducibility
    % eigenmodes decomposition (based on original cPCA paper)
    [V_i,D_i] = eig(C);
    % sort eigenvalues in descending order
    [D_i, ind] = sort(diag(D_i), 'descend');
    V(:,:,alpha_i) = V_i(:,ind(1:min([d_max length(ind)])));
    D(:,alpha_i)   = D_i(1:min([d_max length(ind)])); clear V_i D_i ind
    % calculating intrinsic dimensionality for current alpha
    D(:,alpha_i) = D(:,alpha_i) - min(D(:,alpha_i));
    lambda = D(:,alpha_i)/sum(D(:,alpha_i));
    no_dims_alpha(alpha_i) = 0;
    ind = find(lambda > 0.025);
    if length(ind) > d_max
        no_dims_alpha(alpha_i) = d_max;
    else
        no_dims_alpha(alpha_i) = max([2 length(ind)]);
    end
end
% affinity_matrix
for alpha_i = 1:n_alphas
    for alpha_j = 1:n_alphas
        if alpha_i ~= alpha_j
            theta = subspacea(V(:,1:min([no_dims_alpha(alpha_i) no_dims_alpha(alpha_j)]),alpha_i),...
                V(:,1:min([no_dims_alpha(alpha_i) no_dims_alpha(alpha_j)]),alpha_j));
            affinity_matrix(alpha_i,alpha_j) = prod(cos(theta));
        end
    end
end
affinity_matrix = double(affinity_matrix);
figure; imagesc(affinity_matrix); title('Subspaces affinity matrix'); colorbar; colormap Jet;
% Clustering
rng('default'); Ci = GCSpectralClust1(affinity_matrix,10); Kbst=CNDistBased(Ci,affinity_matrix); Ci = Ci(:,Kbst);
% rng('default'); C = SpectralClustering(affinity_matrix+affinity_matrix', 2, 3); for k = 1:size(C,2), Ci(find(C(:,k)),1) = k; end

Ci(1) = 1; Ci(2:end) = Ci(2:end)+1;
n_clusters = length(unique(Ci));
% computing medoid for each cluster
plot_results = 0;
for clus_i = 1:n_clusters
    ind = find(Ci == clus_i);
    [~,j] = max(sum(affinity_matrix(ind,ind),2));
    Vmedoid(:,:,clus_i) = V(:,:,ind(j));
    Dmedoid(:,clus_i)   = D(:,ind(j));
    alphas_f(clus_i)    = alphas(ind(j));
    no_dims(clus_i)     = no_dims_alpha(ind(j));
    
    % Transforming data, applying mapping on the data
    cPCs(:,:,clus_i)    = X * Vmedoid(:,:,clus_i);
    contrasted_data(:,:,clus_i) = cPCs(:,:,clus_i)*Vmedoid(:,:,clus_i)'*diag(std_data) + mean_data; % see https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com
    
    if plot_results
        figure;  hold on;
        if clus_i == 1
            unique_classes_for_colours = unique(classes_for_colours);
            colours2classes = [1 1 0; ... % yellow
                0 1 1; ... % cyan
                0 1 0; ... % green
                0 0 1; ... % blue
                1 0 1; ... % magenta
                1 0 0; ... % red
                0.5430 0 0]; % dark red
            if length(unique_classes_for_colours) > 7
                disp('Warning: Only 7 different classes (plus background) are considered for the colouring...');
                color_class = colours2classes(end,:);
            end
        end
        for class = 1:length(unique_classes_for_colours)
            ind = find(classes_for_colours == unique_classes_for_colours(class));
            if no_dims(clus_i) < 3
                plot(cPCs(ind,1,clus_i),cPCs(ind,2,clus_i),'.','color',colours2classes(class,:));
            else
                if class > size(colours2classes,1), colours2classes(class,:) = [0.5430 0 0]; end
                plot3(cPCs(ind,1,clus_i),cPCs(ind,2,clus_i),cPCs(ind,3,clus_i),'.','color',colours2classes(class,:));
            end
        end
        if no_dims(clus_i) < 3
            plot(cPCs(indices_background,1,clus_i),cPCs(indices_background,2,clus_i),'.','color',[0 0 0]); % Background in black.
        else
            plot3(cPCs(indices_background,1,clus_i),cPCs(indices_background,2,clus_i),cPCs(indices_background,3,clus_i),'.','color',[0 0 0]); % Background in black.
        end
        title(['cPC, alpha -> ' num2str(alphas_f(clus_i))]);
    end
    
    % Calculating intrinsic dimensionality for current alpha
    lambda = Dmedoid(:,clus_i) ./ sum(Dmedoid(:,clus_i));
    no_dims(clus_i) = 0;
    while no_dims(clus_i) < size(X_background,2) - 1 && lambda(no_dims(clus_i) + 1) > 0.025
        no_dims(clus_i) = no_dims(clus_i) + 1;
    end
    if no_dims(clus_i) < 2
        no_dims(clus_i) = 2;
    end
    
    % Evaluating relative clustering tendency of the intrinsic target data, for this cluster
    dist_matrix = find_nn(cPCs(:,1:no_dims(clus_i),clus_i), ceil(0.01*size(cPCs,1))); dist_matrix = dist_matrix + dist_matrix'; dist_matrix(isnan(dist_matrix)) = 0;
    gap_values(clus_i) = 1/sum(sum(dist_matrix(indices_background,indices_target)))+1/sum(sum(dist_matrix(indices_target,indices_target)));
end
return;