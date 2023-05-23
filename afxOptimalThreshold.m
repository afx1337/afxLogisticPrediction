function [optThr,absDiff] = afxOptimalThreshold(yfit,y,tol,inverse)
    % optimal threhold minimizes sum/mean of absVolDiff
    % this script uses an adaptive/iterative grid search for efficiency
    
    % for debugging purposes
    %load('D:\projects\afxRs\afxLogisticPrediction\tmp_optThr.mat');
    
    gridSize = 10;
    %reshape y: 
    y2 = y(:);
    sumVolGT = sum(y2);
    
    %shape boundaries
    boundaries = [ min(yfit)  max(yfit)];
    while abs(diff(boundaries)) > 2*tol %tol = 0.1
        boundaries = afxFindMinGrid(yfit,sumVolGT,inverse,boundaries,gridSize);
    end
    optThr = round(mean(boundaries),floor(-log10(tol)));
    absDiff = afxAbsDiff(yfit,sumVolGT,optThr,inverse);
end
    
function absDiff = afxAbsDiff(yfit,sumVolGT,threshold,inverse)
    if inverse
        absDiff = sum(abs(sumVolGT-sum(yfit<threshold,1)));
    else
        absDiff = sum(abs(sumVolGT-sum(yfit>threshold,1)));
    end
end

function boundaries = afxFindMinGrid(yfit,sumVolGT,inverse,boundaries,n)
    thrs = boundaries(1):(boundaries(2)-boundaries(1))/(n-1):boundaries(2);
    absDiffs = nan(1,length(thrs));
    for i = 1:length(thrs)
        absDiffs(i) = afxAbsDiff(yfit,sumVolGT,thrs(i),inverse);
    end
    %plot(thrs,absDiffs);
    [~,idx] = min(absDiffs);
    d = thrs(2)-thrs(1);
    boundaries = thrs(idx)+[-d d];
end