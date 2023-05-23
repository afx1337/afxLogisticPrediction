function afxWritePredictors(fname,predictors,meanRSquared)
    fileID = fopen([fname '.txt'],'w');
    fprintf(fileID,'no.  mean(RSquared)  name');
    predictors = [ 'intercept' predictors ];
    for i = 1:length(meanRSquared)
        fprintf(fileID,'\n%03i  %.6f        %s',i,meanRSquared(i),predictors{i});
        info(i).name = predictors{i};
        info(i).meanRSquared = meanRSquared(i);
    end
    fclose(fileID);
    save([fname '.mat'],'info');
end