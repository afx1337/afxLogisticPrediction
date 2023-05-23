function thresholdMaps = afxThresholdModel(x,y,idxTrain,design,space,masks)

    % get tici column
    idxTICI = find(strcmpi(design.predictors,'tici'),1);
    % calculate CBFrel
    x(:,end+1,:) = afxCBFRel(x,y,design,space,masks);
    design.predictors{end+1} = 'CBFRel';
    % get threshold map structure
    thresholdMaps = design.thresholdMaps;
    % obtain threshold maps and optimal thresholds
    for iThreshold = 1:length(design.thresholdMaps)
        idxThr = find(strcmpi(design.predictors,design.thresholdMaps(iThreshold).name),1);
        idxOptThr = idxTrain & x(:,idxTICI,1)' == design.thresholdMaps(iThreshold).tici;
        thresholdMaps(iThreshold).optThr = afxOptimalThreshold(squeeze(x(idxOptThr,idxThr,:)),y(idxOptThr,:),.01,design.thresholdMaps(iThreshold).inverse);
        thresholdMaps(iThreshold).dat = x(:,idxThr,:);
    end
end