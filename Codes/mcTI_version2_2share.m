function [global_pseudotimes,C_all,dim_reduction_outputs,R2values_subtypes,Pvalues_subtypes,R2_nonspecific,P_nonspecific,MDS_coords] = mcTI_version2_2share(Data,starting_point,classes_for_colours,final_subjects,method,max_cPCs,max_CLUs)
% Of note, each subject should have data for at least one modality.
% INPUTS:
% Data: Struct variable, where "Data(modality_i).data" is a numeric matrix
% of size [N_subjects x Nfeatures_modality_i], with each column corresponding to
% a feature of data modality i. For all modalities, use same N_subjects and 
% subjects ordering, filling with NaN for those subjects that don't have a 
% given modality.
% starting_point: column vector, with rows of background subjects.
% classes_for_colours (optional): categorical variable (e.g. diagnoses) just 
% using for plotting with different colors in the cPCA space. Default: [].
% final_subjects (optional): column vector, with rows of target subjects. Default: [].
% method: 'cPCA', 'cPCA with Kernel smoothing' or 'cKPCA'. Recommended: 'cPCA'. 
% max_cPCs: maximum number of cPCAs. Default: 10.
% max_CLUs: maximum number of subtypes. Default: 6.
%
% OUTPUTS:
% global_pseudotimes: [N_subjects x 1] pseudotime values.
% C_all: Subtypes (background subpopulation has value 1 by definition).
% dim_reduction_outputs: Struct variable, where "dim_reduction_outputs(modality_i).Node_contributions" has
% features-specific contributions for modality i.
% R2values_subtypes & Pvalues_subtypes: R2 and P values associated with each subtype 
% (reflecting the subtype specific predictability of pathological advance). 
% R2_nonspecific & P_nonspecific: For reference/comparison, R2 and P values are also 
% provided for the whole no-subtyped population (excluding background subjects).
% MDS_coords: Ignore for now, it is for toolbox.
%-------------------------------------------------------------------------%
% Author: Yasser Iturria medina, 07/07/2022.
% See:
% Iturria-Medina et al., 2022. Science Advances. DOI 10.1126/sciadv.abo6764.
% Iturria-Medina et al., 2020. Brain, 143(2):661-673, https://doi.org/10.1093/brain/awz400.
% Wang et al., 2014, Nature Methods, Vol11, #3. See
% https://www.etriks.org/snf/.

% License/Use ------------------------------------------------------------% 
% Dr. Yasser Iturria-Medina, the NeuroInformatics for Personalized Medicine 
% lab (the NeuroPM-lab), McGill University. 2020.
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
%-------------------------------------------------------------------------%

rng('default'); % for reproducibility
dbstop if error; warning off;
if nargin < 6 || isempty(max_cPCs), max_cPCs = 10; end
if nargin < 7 || isempty(max_CLUs), max_CLUs = 6; end

N_modalities = length(Data);
N_nodes      = size(Data(1).data,1);
if nargin < 4 || isempty(final_subjects), final_subjects = setdiff(1:N_nodes,starting_point)'; else, final_subjects = final_subjects(:); end
try, classes_for_colours = classes_for_colours(1:N_nodes);
catch,
    classes_for_colours(starting_point) = 1;
    if length(final_subjects) < length(setdiff(1:N_nodes,starting_point)) % in case a target subpopulation was defined
        classes_for_colours(setdiff(1:N_nodes,[starting_point(:);final_subjects(:)])) = 2;
        classes_for_colours(final_subjects) = 3;
    else
        classes_for_colours(final_subjects) = 2;
    end
end
        
reduce_dim = 1;
for modl = 1:N_modalities
    data = double(Data(modl).data);
    
    % Finding indices of missing subjects, to deal with missing data without imputation (like NEMO).
    ind_missing_subjects = find(sum(isnan(data),2) == size(data,2));
    ind_present_subjects = find(sum(isnan(data),2) ~= size(data,2));
    data = data(ind_present_subjects,:);    
    [i,j] = ismember(starting_point,ind_present_subjects); i = find(i); starting_point_modl = j(i);     
    [i,j] = ismember(final_subjects,ind_present_subjects); i = find(i); final_subjects_modl = j(i);
    classes_for_colours_modl = classes_for_colours(ind_present_subjects);
    
    % Correcting very-extreme outliers and missing values
    for i = 1:size(data,2), A = data(:,i); A = A(~isnan(A)); data(data(:,i)<median(A)-5*mad(A) | data(:,i)>median(A)+5*mad(A),i) = NaN; end
    try, data = knnimpute(data',10,'Distance','seuclidean')';
    catch, data = TSR(data); end
    
    % Continuing analysis
    data = (data - repmat(mean(data(starting_point_modl,:)),[size(data,1) 1]))./ ...
        repmat(std(data(starting_point_modl,:))+eps,[size(data,1) 1]);
    data = zscore(data);
    if size(data,2) > 0.1*N_nodes || reduce_dim == 1 %--- Reducing dimensionality using cPCA
        disp(['Data modality ' num2str(modl) ', contrastive Dimensionality Reduction...'])
        if strcmp(method,'cPCA')
            [cPCs,gap_values,alphas,no_dims,contrasted_data,Vmedoid,Dmedoid] = contrastivePCA(data,starting_point_modl,final_subjects_modl,max_cPCs,classes_for_colours_modl);
        elseif strcmp(method,'cPCA with Kernel smoothing')
            [cPCs,gap_values,alphas,no_dims,contrasted_data,Vmedoid,Dmedoid] = contrastiveKernelPCA(data,starting_point_modl,final_subjects_modl,max_cPCs,classes_for_colours_modl);
        elseif strcmp(method,'cKPCA')
            [cPCs,gap_values,alphas,no_dims,contrasted_data,Vmedoid,Dmedoid] = cKernelPCA(data,starting_point_modl,final_subjects_modl,max_cPCs,classes_for_colours_modl);
        end
        [~,j]           = max(gap_values); % the optimun alpha should maximizes the clusterization in the target dataset
        data            = cPCs(:,1:no_dims(j),j);
        data            = (data - repmat(mean(data(starting_point_modl,:)),[size(data,1) 1]));
        data            = data/std(data(:));
        % Optional to adjust very-extreme outliers on cPCs.
        for i = 1:size(data,2), A = data(:,i); A = A(~isnan(A)); data(data(:,i)<median(A)-5*mad(A) | data(:,i)>median(A)+5*mad(A),i) = NaN; end
        try, data = knnimpute(data',10,'Distance','seuclidean')'; 
        catch, data = TSR(data); end
        contrasted_data = contrasted_data(:,:,j);
        Node_Weights    = Vmedoid(:,1:no_dims(j),j);
        Lambdas         = Dmedoid(1:no_dims(j),j);
        Node_contributions = (100*(Node_Weights.^2)./repmat(sum(Node_Weights.^2,1),size(Node_Weights,1),1))*Lambdas;
        figure; gscatter(data(:,1),data(:,2),classes_for_colours_modl); title(['Data modality ' num2str(modl) ', cPCs space']); xlabel('cPC1'); ylabel('cPC2');
        disp(['Final # of components (cPCA) -> ' num2str(no_dims(j))]);
        disp(['Final alpha (cPCA) -> ' num2str(alphas(j))]);        
        dim_reduction_outputs(modl).Node_Weights = Node_Weights;
        dim_reduction_outputs(modl).Node_contributions = Node_contributions; clear contrasted_data Node_contributions Node_Weights        
    else
        dim_reduction_outputs(modl).Node_Weights = [];
        dim_reduction_outputs(modl).Node_contributions = [];
    end
    dim_reduction_outputs(modl).mappedX = NaN(N_nodes,size(data,2));
    dim_reduction_outputs(modl).mappedX(ind_present_subjects,:) = data;
    dim_reduction_outputs(modl).ind_present_subjects = ind_present_subjects;
    % Construct similarity graphs
    K = 10;         % Number of neighbors, usually (10~30). Wang et al., 2014.
    alpha = 0.55;   % Hyperparameter, usually (0.3~0.8). Wang et al., 2014.
    T = 30;         % Number of Iterations, usually (10~30). Wang et al., 2014.
    dist_matrix = squareform(pdist(data,'mahalanobis')); 
    W_modl = local_scaling_affinityMatrix(dist_matrix, K);
    if ~isempty(ind_missing_subjects),
        W = NaN([N_nodes N_nodes]);
        W(ind_present_subjects,ind_present_subjects) = W_modl;
    else
        W = W_modl;
    end        
    W3(:,:,modl) = W; 
    % Probability Transition Matrices (more comparable between modalities than original W)
    S = W./repmat(sum(W,2,'omitnan'),[1 N_nodes]);
    W3(:,:,modl) = (S + S')/2; % NEW!!! April 16-2021
    
    eval(['W_all{1,modl} = W;']); clear W_modl W    
    if modl == N_modalities % Assigning mean values to missing modalities.
        W2 = nanmean(W3,3); clear W3
        if sum(isnan(W2(:))), W2 = impute_network(W2); end
        for mold2 = 1:N_modalities
            W = cell2mat(W_all(mold2));            
            ind_missing_subjects = find(isnan(W));
                     S = W./repmat(sum(W,2,'omitnan'),[1 N_nodes]);
                     W = S; % NEW!!! April 16-2021
            W(ind_missing_subjects) = W2(ind_missing_subjects);
            eval(['W_all{1,mold2} = W;']);
        end
    end
end

%--- Node-node distance
disp('Node-node distance and shortest paths ...')
ind_nonstarting = setdiff(1:N_nodes,starting_point);
if N_modalities > 1 % Similarity Network Fusion(SNF):    
    modalities_present = []; for modl = 1:N_modalities, modalities_present(:,modl) = sum(isnan(Data(modl).data),2)<0.5*size(Data(modl).data,2); end
    figure; imagesc(modalities_present([starting_point(:); final_subjects(:)],:)); colorbar; title('Present modalities (only background & target)')
    similarity = SNF(W_all, K, T);
    similarity = similarity - diag(diag(similarity)); similarity = similarity/max(similarity(:)); similarity = similarity + eye(size(similarity));
    figure; imagesc(similarity([starting_point(:); final_subjects(:)],[starting_point(:); final_subjects(:)])); colorbar; title('Fused Subject-Subject Similarity Network (only background & target)');
    dist_matrix = (1./similarity); dist_matrix = dist_matrix - diag(diag(dist_matrix)); dist_matrix(isinf(dist_matrix) | isnan(dist_matrix)) = 0;
else
    similarity  = cell2mat(W_all(1)); similarity = similarity - diag(diag(similarity)); similarity = similarity/max(similarity(:)); similarity = similarity + eye(size(similarity));
    figure; imagesc(similarity([starting_point(:); final_subjects(:)],[starting_point(:); final_subjects(:)])); title('Subject-Subject Similarity Network (background & target)');
    dist_matrix = (1./similarity); dist_matrix = dist_matrix - diag(diag(dist_matrix)); dist_matrix(isinf(dist_matrix) | isnan(dist_matrix)) = 0;
end

%--- Minimal spanning tree across all the points
% Specifying which node is the root, the closest one to all the starting points
[~,j]     = max(mean(similarity(starting_point,starting_point))); 
Root_node = starting_point(j);

% Calculating minimum spanning tree
in_background_target  = [starting_point(:); final_subjects(:)];
out_background_target = setdiff(1:N_nodes,in_background_target)';
if length(final_subjects) < length(setdiff(1:N_nodes,starting_point)) % in case a target subpopulation was defined
   dist_matrix0 = dist_matrix;  similarity0 = similarity; 
   dist_matrix  = dist_matrix(in_background_target,in_background_target); % only considering background and target populations
   similarity   = similarity(in_background_target,in_background_target);
   Root_node    = j;
end
rng('default'); % For reproducibility
Tree = graphminspantree(sparse(dist_matrix),Root_node);
Tree(Tree > 0) = dist_matrix(Tree > 0);
MST = full(Tree + Tree'); clear Tree d

%--- Shortest paths to the starting point(s)
option_paths = 1;
if option_paths == 1,
    % Based on minimum spanning tree
    [dist,~,pred] = graphshortestpath(sparse(MST),Root_node(1)); dist(dist(:)~=0) = dist(dist(:)~=0)-min(dist(dist(:)~=0)); datas.A = dist(:); datas.F = pred(:); clear dist pred
elseif option_paths == 2,    
    % Based on original node-node similarity
    [dist,~,pred] = graphshortestpath(sparse(dist_matrix),Root_node(1)); dist(dist(:)~=0) = dist(dist(:)~=0)-min(dist(dist(:)~=0)); datas.A = dist(:); datas.F = pred(:); clear dist pred
end
max_distance = max(datas.A(~isinf(datas.A))); 

if length(final_subjects) < length(setdiff(1:N_nodes,starting_point)) % in case a target subpopulation was defined
    ind_disconnected = find(isinf(datas.A));
    if ~isempty(ind_disconnected) % may happens when defining a selected target population.
        for node_i = ind_disconnected(:)'
            [~,j] = sort(similarity0(in_background_target(node_i),:),'descend');
            for node_j = j(:)'
                [~,j2] = sort(similarity0(node_j,in_background_target),'descend');
                ind = find(~isinf(datas.A(j2)));
                if ~isempty(ind), datas.A(node_i) = datas.A(j2(ind(1))); break; end
            end
        end
    end    
    global_pseudotimes(in_background_target,1) = datas.A(:)/max_distance;
    % Version 3 (added on July 7th, 2022) - weighted average of top 5 nearest neighbors
    temp_sim = similarity0(out_background_target,in_background_target);
    for i = 1:length(out_background_target), [~,j] = sort(temp_sim(i,:),'descend'); temp_sim(i,j(5+1:end)) = 0; end
    global_pseudotimes(out_background_target,1) = (temp_sim*global_pseudotimes(in_background_target))./sum(temp_sim,2); clear temp_sim; % Based in weigthed average
    % Continuing
    dist_matrix = dist_matrix0; similarity = similarity0;
else
    global_pseudotimes(:,1) = datas.A(:)/max_distance;
end
[~,global_ordering] = sort(global_pseudotimes);

try
    figure; hold on; title('Pseudo-times in background and target populations')
    try
        for ii=1:2, % only for the non-distributed toolbox version (i.e. for lab use, because notBoxPlot its GNU license)
            if ii == 1, notBoxPlot(global_pseudotimes(starting_point),ii,'jitter',0.5,'style','patch');
            else, notBoxPlot(global_pseudotimes(final_subjects),ii,'jitter',0.5,'style','patch'); end; % GNU LICENSE https://github.com/raacampbell/notBoxPlot/blob/master/LICENSE
        end
    catch
        group = zeros(length(starting_point)+length(final_subjects),1); 
        group(1:length(starting_point),1) = 1; group(length(starting_point)+1:end,1) = 2;
        boxplot(global_pseudotimes([starting_point(:); final_subjects(:)]),group,'Notch','on','Whisker',1,'BoxStyle','outline');
    end
    xticks(1:2); xticklabels({'Background'; 'Target'}); set(gca,'XTickLabelRotation',45);
end

% Multidimensional Scaling to convert integrated distance network to coordinates
rng('default'); opts = statset('Display','iter'); clear MDS_coords stress temp;
dist_matrix = dist_matrix/(max(dist_matrix(:))+eps); dist_matrix(dist_matrix==0) = max(dist_matrix(:)); dist_matrix = dist_matrix - diag(diag(dist_matrix));
% % Paper mcTI version 1 (before July 7-2022)
% MDS_coords = mdscale(dist_matrix,max([3 max_CLUs]),'criterion','metricsstress','Options',opts);
% Paper mcTI version 2 (added on July 7-2022)
% Using BIC to select MDS dimensions.
select_BIC_MDS = 1;
if select_BIC_MDS
    clear BIC_MDS
    for i = 2:10 % Calculating before-subtyping predictability as a function of DMS dimension
        disp(['MDS dimensions -> ' num2str(i)])        
        MDS_coords = mdscale(dist_matrix,i); % MDS_coords = mdscale(dist_matrix,i,'criterion','metricsstress'); % MDS_coords = cmdscale(dist_matrix,i);
        X = MDS_coords; Y = global_pseudotimes;
        B = [ones(length(Y),1) X]\Y;
        Yhat = [ones(length(Y),1) X]*B;
        k = length(B); residual = (Y - Yhat)'; sigma_Y = std(Y);
        L = (1/(sigma_Y*sqrt(pi)))*exp(-(residual*residual')/(2*sigma_Y^2));
        BIC_MDS(i) = k*log(length(Y)) - 2*log(L); % https://en.wikipedia.org/wiki/Bayesian_information_criterion
        [R_nonspecific,P_nonspecific] = corr(Y,Yhat); R2_nonspecific = 100*(R_nonspecific^2);
        figure; plot(Y,Yhat,'.'); title(['No Subtyped, MDS dimensions = ' num2str(i) ', R2 = ' num2str(R2_nonspecific)])
    end
    [~,j] = min(BIC_MDS(2:end)); opt_DMS_dim = j + 1;
    disp(['Optimum DMS-dim -> ' num2str(opt_DMS_dim)])    
    MDS_coords = mdscale(dist_matrix,opt_DMS_dim); 
else
    MDS_coords = mdscale(dist_matrix,max([3 max_CLUs]));
end
u_classes_for_colours = unique(classes_for_colours);
colours2classes = [1 1 0; 0 1 1; 0 1 0; 0 0 1; 1 0 1; 1 0 0; 0.5430 0 0];... % yellow cyan green blue magentared dark dark-red.
figure; hold on; title(['Final fused dissimilarity space (MDS)']); xlabel('coord1'); ylabel('coord2'); ylabel('coord3'); 
for i = 1:length(u_classes_for_colours),
    try, plot3(MDS_coords(classes_for_colours==u_classes_for_colours(i),1),...
        MDS_coords(classes_for_colours==u_classes_for_colours(i),2),...
        MDS_coords(classes_for_colours==u_classes_for_colours(i),3),'o','color',colours2classes(min([i 7]),:)); end; end

%--- Final clustering of sub-trajectories 
Predecessors = datas.F';
ind_in_paths = find(Predecessors ~= -1);
Tails = 1:N_nodes; % Taking/clustering all nodes as "Tails" (and not just connected nodes that are not Predecessors)
S     = similarity;

% Our E-M clustering
BIC(1) = Inf; clear Clust
for k = 2:max_CLUs
    rng('default');
    C_k = GCSpectralClust1(S(ind_nonstarting,ind_nonstarting),k); C_k = C_k(:,end);
    [Clust(:,k),~,R2_targetclusters_pred,P_targetclusters_pred,BIC(k)] = ...
        clustering_regress_EM(global_pseudotimes(ind_nonstarting),MDS_coords(ind_nonstarting,:),C_k,500); % with repetition 1, we would keep instead the inputed subtypes (not refined), but calculate their configuration's BIC.
    R2values_subtypes(k,1:length(R2_targetclusters_pred)) = R2_targetclusters_pred;
    Pvalues_subtypes(k,1:length(P_targetclusters_pred))   = P_targetclusters_pred;
end
figure; stem(BIC); title('Bayesian information criterion vs # of subtypes');
[~,Kbst] = min(BIC); % the real number is length(unique(Clust(:,Kbst)))
R2values_subtypes = R2values_subtypes(Kbst,1:length(unique(Clust(:,Kbst))));
Pvalues_subtypes  = Pvalues_subtypes(Kbst,1:length(unique(Clust(:,Kbst)))); % P-values for non-background subtypes
C_orig(starting_point,1) = 1; C_orig(ind_nonstarting) = Clust(:,Kbst) + 1;

% Optional statistical stability analysis of Subtypes
calculate_subtypes_significance = 0;
if calculate_subtypes_significance
    % Cluster stability analysis
    rng('default'); N_permutations = 50; N_randomizations = 100*N_permutations; 
    Stemp = S(ind_nonstarting,ind_nonstarting);
    pairwise_stability_rate = zeros(size(Stemp,1),size(Stemp,1),max_CLUs); 
    subject_subject         = zeros(size(Stemp,1),size(Stemp,1)); clear C_orig Concordance;
    h = waitbar(0,'Bootstrapping to evaluate subtypes stability...');
    for perm = 1:N_permutations
        waitbar(perm / N_permutations);
        ind_present_subjects = randperm(size(Stemp,1)); 
        ind_present_subjects = ind_present_subjects(1:round(0.95*size(Stemp,1))); % Bootstrapping, taking only 95% of the sample (without replacement).
        Crep = GCSpectralClust1((Stemp(ind_present_subjects,ind_present_subjects)+Stemp(ind_present_subjects,ind_present_subjects)')/2,Kbst);
        [Crep,~,R2_targetclusters_pred,P_targetclusters_pred,BIC(k)] = ...
        clustering_regress_EM(global_pseudotimes(ind_nonstarting),MDS_coords(ind_nonstarting,:),C_k,500); % with repetition 1, we would keep instead the inputed subtypes (not refined), but calculate their configuration's BIC.
        for k = 2:max_CLUs
            Concordance(k,perm) = Cal_NMI(Ci(ind_present_subjects,k), Crep(:,k));
            for k2 = 1:k, pairwise_stability_rate(ind_present_subjects(Crep(:,k)==k2),ind_present_subjects(Crep(:,k)==k2),k) = ...
                    pairwise_stability_rate(ind_present_subjects(Crep(:,k)==k2),ind_present_subjects(Crep(:,k)==k2),k) + 1;
            end
        end
        subject_subject(ind_present_subjects,ind_present_subjects) = subject_subject(ind_present_subjects,ind_present_subjects) + 1;
    end
    close(h);  
    
    pairwise_stability_rate_null = zeros(size(S,1),size(S,1),max_CLUs,N_randomizations/N_permutations);
    subject_subject_null         = zeros(size(S,1),size(S,1));
    h = waitbar(0,'Random permutations to calculate subtypes significance...');
    for perm_null = 1:N_randomizations
        waitbar(perm_null / N_randomizations);
        ind_present_subjects = randperm(size(S,1)); ind_present_subjects = ind_present_subjects(1:round(0.95*size(S,1)));
        S_rand = zeros(size(S,1),size(S,1)); ind = find(triu(ones(size(S,1),size(S,1)))); S_rand(ind) = S(ind(randperm(length(ind)))); S_rand = S_rand + S_rand';
        Crep = GCSpectralClust1((S_rand(ind_present_subjects,ind_present_subjects)+S_rand(ind_present_subjects,ind_present_subjects)')/2,max_CLUs);
        for k = Kbst % 1:max_CLUs
            for k2 = 1:k
                pairwise_stability_rate_null(ind_present_subjects(Crep(:,k)==k2),ind_present_subjects(Crep(:,k)==k2),k,floor((perm_null-1)/N_permutations)+1) = ...
                    pairwise_stability_rate_null(ind_present_subjects(Crep(:,k)==k2),ind_present_subjects(Crep(:,k)==k2),k,floor((perm_null-1)/N_permutations)+1) + 1;
            end
        end
        subject_subject_null(ind_present_subjects,ind_present_subjects) = ...
            subject_subject_null(ind_present_subjects,ind_present_subjects) + 1;
    end
    close(h);
    Pvalues_subtypes = NaN(max_CLUs,max_CLUs); clear avr_stability avr_stability_null
    for k = Kbst % 1:max_CLUs,
        disp(k)
        for k2 = 1:k
            temp = pairwise_stability_rate(Ci(:,k)==k2,Ci(:,k)==k2,k)./subject_subject(Ci(:,k)==k2,Ci(:,k)==k2);
            avr_stability(k,k2) = mean(temp(:));
            if calculate_subtypes_significance
                for k3 = 1:N_randomizations/N_permutations
                    temp = pairwise_stability_rate_null(Ci(:,k)==k2,Ci(:,k)==k2,k,k3)./subject_subject_null(Ci(:,k)==k2,Ci(:,k)==k2);
                    avr_stability_null(k,k2,k3) = mean(temp(:));
                end
                Pvalues_subtypes(k,k2) = 1 - sum(avr_stability(k,k2) > avr_stability_null(k,k2,:))/(N_randomizations/N_permutations);
            end
        end
    end    
end

%--- Assigning trajectories belonging to each cluster
comms = unique(C_orig(:)); N_comms = length(comms); 
disp('Identifying final trajectories ...')
counter = 0; clear TRAJ N_temp Probs Yhat;
for comm_i = comms(:)'
    counter = counter + 1;
    ind0 = Tails(C_orig == comm_i);
    ind  = ind0; % for considering all the trajectories inside one cluster
    
    p = [];
    TRAJ(counter).points = ind(:)';
    % Adding nodes connected (in their path) to these points:
    [i,j] = ismember(ind(:),in_background_target(:));
    indj  = j(i);
    for ind_i = indj(:)'
        node_target = ind_i; curv = node_target; p = [p, curv];
        % Towards the origin
        while Predecessors(curv) > 0
            curv = Predecessors(curv); p = [p, curv]; end
    end
    TRAJ(counter).points = unique([TRAJ(counter).points(:); in_background_target(p(:))]); clear curv
    [i2,j2] = sort(global_pseudotimes(TRAJ(counter).points));
    TRAJ(counter).points             = TRAJ(counter).points(j2);
    TRAJ(counter).reduced_trajectory = MDS_coords(TRAJ(counter).points,:);
    TRAJ(counter).distances          = global_pseudotimes(TRAJ(counter).points)*max_distance;
    TRAJ(counter).pseudo_times       = global_pseudotimes(TRAJ(counter).points);
    
    % Estimating each node's probablity to belong to this trajectory (using
    % the adjusted-MDS coordinates)
    adjMDS_traj = MDS_coords(TRAJ(counter).points,:);
    for i = 1:size(MDS_coords,2)
        B = [ones(length(j2),1) global_pseudotimes(TRAJ(counter).points)]\MDS_coords(TRAJ(counter).points,i); % for adjusting by global_pseudotimes.
        adjMDS_traj(:,i) = MDS_coords(TRAJ(counter).points,i) - [ones(length(j2),1) global_pseudotimes(TRAJ(counter).points)]*B;
    end
    try
        Probs(TRAJ(counter).points,counter) = mvnpdf(adjMDS_traj,mean(adjMDS_traj),cov(adjMDS_traj));
    catch
        Probs(TRAJ(counter).points,counter) = 0;
    end
end
Probs_all = Probs./repmat(sum(Probs,2),[1 size(Probs,2)]);
figure; imagesc(Probs_all); colormap Jet; colorbar; title('Individual probabilities to belong to each identified sub-trajectory');
ylabel('Subjects'); xlabel('m-cTI Sub-trajectories');

% Paper mcTI version 2 (added on July 7-2022)
C_all = C_orig; % output from the EM algorithm (as described in paper)

presence_matrix  = zeros(length(unique(classes_for_colours(:))),length(TRAJ));
presence_matrix2 = zeros(length(unique(classes_for_colours(:))),length(unique(C_all)));
counter = 0; 
for value = unique(classes_for_colours(:))'
    counter = counter + 1;
    for i = 1:length(TRAJ),
        presence_matrix(counter,i) = 100*length(find(classes_for_colours(TRAJ(i).points) == value))/length(find(classes_for_colours == value));
    end
    for i = 1:length(unique(C_all))
        presence_matrix2(counter,i) = 100*length(find(classes_for_colours(C_all==i) == value))/length(find(classes_for_colours == value));
    end
end
figure; imagesc(presence_matrix); colormap Jet; title('% of Real groups (e.g. Background, Target) on identified Sub-trajectories');
for i = 1:size(presence_matrix,2)
  for j = 1:size(presence_matrix,1)
      text(i,j,num2str(presence_matrix(j,i),2));
  end
end
ylabel('Data sub-groups'); xlabel('cTI Sub-trajectories');
figure; imagesc(presence_matrix2); colormap Jet; title('% of Real groups (e.g. Background, Target) on identified Subtypes');
for i = 1:size(presence_matrix2,2)
  for j = 1:size(presence_matrix2,1)
      text(i,j,num2str(presence_matrix2(j,i),2));
  end
end
ylabel('Data sub-groups'); xlabel('cTI Subtypes');

% Calculating-Plotting within-subtype predictability
rng('default');
for i = 1:length(unique(C_all))-1
    ind = find(C_all == i+1);
    X = MDS_coords(ind,:); Y = boxcox(global_pseudotimes(ind));
    B = [ones(length(Y),1) X]\Y;
    Yhat = [ones(length(Y),1) X]*B;
    [R,P] = corr(Y,Yhat);
    R2values_subtypes(i) = 100*R^2; Pvalues_subtypes(i)  = P;
    figure; plot(Y,Yhat,'.'); title(['Subtype -> ' num2str(i) ', R2 = ' num2str(R2values_subtypes(i))])
end
% Calculating-Plotting before-subtyping predictability
ind = find(C_all > 1);
X = MDS_coords(ind,:); Y = boxcox(global_pseudotimes(ind));
B = [ones(length(Y),1) X]\Y;
Yhat = [ones(length(Y),1) X]*B;
[R_nonspecific,P_nonspecific] = corr(Y,Yhat); R2_nonspecific = 100*(R_nonspecific^2);
figure; plot(Y,Yhat,'.'); title(['No Subtyped, R2 = ' num2str(R2_nonspecific)])
return;

function [Similarity_imp,R2_pred,alpha_opt] = impute_network(Similarity)
% Based on Ahmad, et al. Missing Link Prediction using Common Neighbor and Centrality 
% based Parameterized Algorithm. Sci Rep 10, 364 (2020). 
% https://doi.org/10.1038/s41598-019-57304-y

disp(['Network re-construction because missing values...'])

N_nodes    = size(Similarity,1);
Similarity = Similarity - diag(diag(Similarity));    
BackBone   = network_backbone1(Similarity, max([4 round(0.01*N_nodes)]));
Dist       = BackBone.^(-1); Dist(isinf(Dist) | isnan(Dist)) = 0;
D          = dijk(Dist); ind = isinf(D); D(ind) = 0; D = N_nodes*D/max(D(:));

ind_all   = find(triu(ones([N_nodes N_nodes]),1));
ind_nonan = find(~isnan(Similarity) & triu(ones([N_nodes N_nodes]),1));
[x_all,y_all] = ind2sub([N_nodes N_nodes],ind_all);
alphas     = [0:0.05:1];
Similarity_score = zeros([N_nodes N_nodes length(alphas)]); Map1 = zeros(N_nodes,N_nodes); Map2 = zeros(N_nodes,N_nodes); clear R
counter    = 0; 
for alpha = alphas % To find optimum alpha
    counter = counter + 1;
    if counter == 1,
        for i_all = 1:length(ind_all);
            vx = find(BackBone(x_all(i_all),:)); vy = find(BackBone(:,y_all(i_all))'); vxy = intersect(vx,vy);
            Map1(x_all(i_all),y_all(i_all)) = length(vxy);
            Map2(x_all(i_all),y_all(i_all)) = N_nodes/D(ind_all(i_all));
        end
    end
    Similarity_score(:,:,counter) = alpha*Map1 + (1-alpha)*Map2;
    temp = Similarity_score(:,:,counter);
    R_pred(counter) = corr(Similarity(ind_nonan),temp(ind_nonan));
end
[i,j]     = max(R_pred);
alpha_opt = alphas(j); R2_pred = 100*R_pred(j)^2;
disp(['Network re-construction accuracy -> ' num2str(R2_pred) ' %'])

Similarity_imp = Similarity_score(:,:,j);
Similarity_imp = Similarity_imp + Similarity_imp';
Similarity_imp = max(Similarity(ind_nonan))*Similarity_imp/max(Similarity_imp(:)); clear Similarity_score;
return;