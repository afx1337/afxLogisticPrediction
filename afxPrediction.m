function [patientsTest,mRSquared] = afxPrediction(x,y,masks,space,design,idxTrain,idxTest)
    
    % fit and save GLM (training data)
    [stats,scale] = afxLogisitcGLMfit(x(idxTrain,:,:),y(idxTrain,:));
    mRSquared = afxSaveModel(stats,masks,space,scale,afxPartialDesign(design,idxTrain));
    % generate design matix for sucessfull and unsucessful reka and
    % predict (it's complicated, becaus all interactions need to be
    % recalculated in the design matrix)
    idxTICI = find(strcmpi(design.predictors,'tici'),1);
    xReka = x(idxTest,1:(length(design.predictors)-length(design.interactions)),:);
    xReka(:,idxTICI,:) = 0;
    [xReka,~] = afxAddInteractions(xReka,design);
    predictions.reka0 = afxLogisticGLMval([stats.beta],xReka,scale);
    xReka = x(idxTest,1:(length(design.predictors)-length(design.interactions)),:);
    xReka(:,idxTICI,:) = 1;
    [xReka,~] = afxAddInteractions(xReka,design);
    predictions.reka1 = afxLogisticGLMval([stats.beta],xReka,scale);
    clear xReka;
    % calculate threshold (min abs.vol.diff.)
    yfit = afxLogisticGLMval([stats.beta],x(idxTrain,:,:),scale);
    optThr = afxOptimalThreshold(yfit,y(idxTrain,:),.01,false);
    % save predictions
    patientsTest = afxSavePredictions(predictions,y(idxTest,:),masks,space,afxPartialDesign(design,idxTest),optThr);
    % threshold variant
    thresholdMaps = afxThresholdModel(x,y,idxTrain,design,space,masks);
    designTest = design; designTest.patients = patientsTest;
    for iThrMap = 1:length(thresholdMaps)
        thresholdMaps(iThrMap).dat = thresholdMaps(iThrMap).dat(idxTest,:);
    end
    patientsTest = afxSaveThreshold(thresholdMaps,masks,space,designTest);
end

function design = afxPartialDesign(design,idx)
    design.patients = design.patients(idx);
end
