function data = TSR(data_missing)
% TSR: Trimmed Scores Regression algorithm for data imputation.
% data_missing = N_obs*N_features array of input data (with NaN for missing values)
% data         = N_obs*N_features array of output data.
% For Trimmed Scores Regression (TSR) description, see original article:
% Folch-Fortuny, A., Villaverde, A.F., Ferrer, A. et al. Enabling network inference methods 
% to handle missing data and outliers. BMC Bioinformatics 16, 283 (2015). 
% https://doi.org/10.1186/s12859-015-0717-7

B = corr(data_missing); B(isnan(B)) = 0;
[~,S,~] = svd(B);
lambdas = diag(S);
npc_max = find(cumsum(lambdas)/sum(lambdas) > 0.9); % maximum number of PCs.

% Detecting local missing data...
[nobs,nfeats] = size(data_missing);
present   = ~isnan(data_missing);
missing   = isnan(data_missing);
data_mean = mean(data_missing,'omitnan');
[x y]     = find(isnan(data_missing));
nmissing  = length(x);

% Adding corresponding mean
data  = data_missing;
for i = 1:nmissing, data(x(i),y(i)) = data_mean(y(i)); end

N_reps  = 1500; convergence = 1.0e-8; updated_convergence = Inf;
counter = 0;
h = waitbar(0,'Imputation, predicting values...');
while updated_convergence > convergence && counter < N_reps % repeat steps until convergence, i.e. when the mean difference between the predicted values for the missing data in step n and n-1 is lower than a specified threshold.
  counter = counter + 1;
  waitbar(counter/N_reps);
  
  datamean = mean(data);
  S        = cov(data);
  data_centered = data - ones(nobs,1)*data_mean;
  if nobs > nfeats
      [U D V]=svd(data_centered,0); 
  else
      [V D U]=svd(data_centered',0);
  end

  PCs = V(:,1:npc_max); % taking the principal components
  for i = 1:nobs,
    if size(data_missing,2) > 0 % # missing values
      PCs_i = PCs(present(i,:),1:min(npc_max,size(present(i,:),2)));
      temp  = data_centered(i,present(i,:))';
      yhat  = S(missing(i,:),present(i,:))*PCs_i*pinv(PCs_i'*S(present(i,:),present(i,:))*PCs_i)*PCs_i'*temp; % fitting a regression model between x#i and the scores of a new PCA model on x?i.
      data_centered(i,missing(i,:)) = yhat'; % incorporating the estimated missing values in the dataset X.
    end
  end
  previous_mis = data(missing(i,:));
  data = data_centered + ones(nobs,1)*data_mean; % Adding the mean of the original variables.
  updated_convergence = mean((data(missing(i,:)) -  previous_mis).^2);
end
close(h);
return;