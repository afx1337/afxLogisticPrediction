function [stats,scale] = afxLogisitcGLMfit(x,y)
    % [stats,scale] = afxLogisitcGLM(x,y)
    %
    % x          ... predictors (observations x predictors x voxels)
    % y          ... response (observations x voxels)
    %
    % stats.beta ... betas
    % stats.t    ... t-values
    % stats.dfe  ... degrees of freedom
    % stats.sse  ... sum of squared errors
    % scale      ... scale factors for x (scale.mean, scale.std)
    %
    % NaNs in x or y are treated as missing values

    s = tic;
    % fit logistic glm for every voxel
    fprintf('Fitting logistic GLMs [');
    ws = warning('off');
  
    % reshape input data (x, y) and clear x,y (for freeing memory)
    y2 = y(:);
    clear y;
    x2 = reshape(permute(x,[2 1 3]),size(x,2),size(x,1)*size(x,3));
    clear x;
    x2 = x2';
    xnan = any(isnan(x2),2);
    
    % delete observations with NaNs
    x2(xnan,:) = [];
    y2(xnan) = [];
    
    % random stratified undersampling
    idxOverweigt = y2==round(mean(y2));
    nDel = nnz(idxOverweigt)-nnz(~idxOverweigt);
    idxOverweigt = find(idxOverweigt);
    idxOverweigt = idxOverweigt(randperm(nnz(idxOverweigt)));
    idxDel = idxOverweigt(1:nDel);
    x2(idxDel,:) = [];
    y2(idxDel) = [];
    
    % scale input data
    scale.mean = nanmean(x2,1);
    scale.std = nanstd(x2,1);
    x2 = (x2-scale.mean)./scale.std;
    
    % fit GLM
    [b,~,statistics] = glmfit(x2,y2,'binomial','link','logit');
    stats = struct('t',{statistics.t},'beta',{b},'dfe',{statistics.dfe},'mse',{nanmean(statistics.resid.^2)});

    warning(ws);
    fprintf('] (%.2f min)\n',toc(s)/60);
end