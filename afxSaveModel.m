function [meanRSquared] = afxSaveModel(stats,masks,space,scale,design)
    
    % file prefix
    prefix = 'LogisticGLM_';

    % destination directory
    destDir = fullfile(design.dataDir,'output',strcat(design.analysisName,'-s',num2str(design.FWHM)),'models');
    if isfield(design,'fold') && design.fold > 0
        destDir = fullfile(destDir,sprintf('fold%03i',design.fold));
    end
    mkdir(destDir);
    
    % write masks
    maskNames = fieldnames(masks);
    for i = 1:length(maskNames)
        afxVolumeWrite(fullfile(destDir,strcat(prefix,'mask_',maskNames{i},'.nii')),masks.(maskNames{i}),space.dim,'int16',space.mat);
    end

    % write model/stats
    save(fullfile(destDir,strcat(prefix,'stats.mat')),'stats');
    
    % save other data (disign, meanRSquared, scale)
    save(fullfile(destDir,strcat(prefix,'design.mat')),'design');
    save(fullfile(destDir,strcat(prefix,'scale.mat')),'scale');
    % save info/meanRSquares
    meanRSquared = nanmean([stats.t].^2./([stats.t].^2+[stats.dfe]),2);
    afxWritePredictors(fullfile(destDir,strcat(prefix,'meanRSquared')),design.predictors,meanRSquared)
end