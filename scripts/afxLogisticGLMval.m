function [yfit] = afxLogisticGLMval(betas,x,scale)
    % [yfit] = afxLogisticGLMval(betas,x,scale)
    %
    % betas ... model parameters
    % x     ... predictors (observations x predictors x voxels)
    % scale ... scale factors to scale x
    %
    % predicts responses based using logistic GLM
    % yfit  ... predicted responses (observations x voxels)
    
     % reshape input data (x)  and clear x (for freeing memory)
    x2 = reshape(permute(x,[2 1 3]),size(x,2),size(x,1)*size(x,3));
    clear x;
    x2 = x2';
    xnan = any(isnan(x2),2);
    
    % delete observations with NaNs
    x2(xnan,:) = [];
    
    % scale x
    if ~isempty(scale)
        x2 = (x2-scale.mean)./scale.std;
    end
    
    % initialize yfit
    yfit = nan(size(x2,1),1);
    % predict response
    yfit = glmval(betas,x2,'logit');
end
