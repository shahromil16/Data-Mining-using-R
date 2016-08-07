function [clusterParameters, estimatedLabels, logLikelihood, costVsComplexity] = ...
    ExMax(inputData, numOfClusters, stopTolerance, numberOfRuns)
 
% INPUTS:
% inputData: nSamples x dDimensions array with the data to be clustered
% numberOfClusters: number of cluster for algorithm
% stopTolerance: parameter for convergence criteria
% numberOfRuns: number of times the algorithm will run with random initial-izations
%
% OUTPUTS:
% clusterParameters: numberOfClusters x 1 struct array with the Gaussian mixture parameters:
%     .mu - mean of the Gaussian component
%     .covariance - covariance of the Gaussian component
%     .prior - prior of the Gaussian component
% estimatedLabels: nSamples x 1 vector with labels based on maximum prob- ability. 
% Since EM is a soft clustering algorithm, its output are just densities for each cluster in the mixture model
% logLikelihood: 1 x numberOfIterations vector with the log-likelihood as a function of iteration number
% costVsComplexity: 1 x maxNumberOfClusters vector with BIC criteria as a function of number of clusters (more on this below)
 
%% Initialize:
logLikelihood = zeros(1,numberOfRuns);
n = size(inputData,2);
k = numOfClusters;
estimatedLabels = ceil(k*rand(1,n));
R = full(sparse(1:n,estimatedLabels,1,n,k,n));
ii = 1;
err = 1;
X = inputData;
C = rand(k,3);
xmin = min(X(1,:));
xmax = max(X(1,:));
ymin = min(X(2,:));
ymax = max(X(2,:));
 
while err > stopTolerance && ii < numberOfRuns
    fprintf('Iteration No: %d\n', ii);
    ii = ii + 1;
    [~,estimatedLabels(1,:)] = max(R,[],2);
    clusterParameters = MSTEP(inputData,R);
    [R, logLikelihood(ii)] = ESTEP(inputData,clusterParameters);
    err = abs(logLikelihood(ii)-logLikelihood(ii-1));
    
    figure(1);
    xlim([xmin xmax]);
    ylim([ymin ymax]);
    titstr = sprintf('k = %d', k);
    title(titstr);
    lb = sort(unique(estimatedLabels));
    
    for i=1:length(lb)
        dataCut = X(:,estimatedLabels==lb(i));
        h=plot(dataCut(1,:),dataCut(2,:),'.',clusterParameters.mu(1,i),clusterParameters.mu(2,i),'*');
        
        set(h(1),'color',C(i,:));
        set(h(2),'linewidth',10,'color',C(i,:));
        hold on;
        
        pause(0.01);
    end
    hold off;
 
    temp = logLikelihood;
    logLikelihood = temp(~isinf(temp));
    costVsComplexity = logLikelihood - (3*k-1)/2*log(length(X));
    
end
 
 
%% E-Step
function [R, logLikelihood] = ESTEP(X, model)
mu = model.mu;
Sigma = model.Sigma;
w = model.w;
[d,n] = size(X);
k = size(mu,2);
R = zeros(n,k);
for i = 1:k
    Xcut = bsxfun(@minus,X,mu(:,i));
    a = dot(pinv(chol(Sigma(:,:,i))')*Xcut,pinv(chol(Sigma(:,:,i))')*Xcut,1);
    b = d*log(2*pi)+2*sum(log(diag(chol(Sigma(:,:,i)))));
    R(:,i) = -(b+a)/2 + log(w(i));
end
y = max(R,[],2);
for i=1:k
    temp(:,i) = R(:,i) - y(i);
end
s = y+(sum(exp(temp),2));
i = isinf(y);
if any(i(:))
    s(i) = y(i);
end
logLikelihood = sum(s)/n;
R = exp(bsxfun(@minus,R,s));
 
%% M-Step
function model = MSTEP(X, R)
[d,n] = size(X);
[~,k] = size(R);
nk = sum(R,1);
w = nk/n;
for i=1:k
    mu(:,i) = X*R(:,i)/(nk(i));
end
S = min(var(X'));
del = S*10e-4;
Sigma = zeros(d,d,k);
Rroot = sqrt(R);
for i = 1:k
    Xcut = bsxfun(@times,bsxfun(@minus,X,mu(:,i)),Rroot(:,i)');
    Sigma(:,:,i) = Xcut*Xcut'/nk(i)+eye(d)*(1e-6);
    % Dealing with singular covariance matrices
    if mean2(isnan(Sigma(:,:,i))) > 0
        Sigma(:,:,i) = del*eye(d);
        EIG = eig(Sigma(:,:,i));
        if length(find(EIG==0))<d
            Sigma(:,:,i) = del*eye(d);
        end
    end
end
 
model.mu = mu;
model.Sigma = Sigma;
model.w = w;
