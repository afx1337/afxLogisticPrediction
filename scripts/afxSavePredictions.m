function [patients] = afxSavePredictions(predictions,y,masks,space,design,optThr)
   
    % file prefix
    prefix = 'LogisticGLM_';

    % destination directory
    destDir = fullfile(design.dataDir,'output','glm',strcat(design.analysisName,'-s',num2str(design.FWHM)),'predictions');
    
    predNames = fieldnames(predictions);
    for iPatient = 1:length(design.patients)
        % patient dir
        patDestDir = fullfile(destDir,design.patients(iPatient).name);
        mkdir(patDestDir);
        % save predictions and prediction mask
        for iPrediction = 1:length(predNames)
            fname = sprintf('%sprediction_%s.nii',prefix,predNames{iPrediction});
            design.patients(iPatient).predictions.glm(iPrediction).name = predNames{iPrediction};
            design.patients(iPatient).predictions.glm(iPrediction).file = afxVolumeWrite(fullfile(patDestDir,fname),afxDeMask(masks.analysis,predictions.(predNames{iPrediction})(iPatient,:)),space.dim,'int16',space.mat);
            design.patients(iPatient).predictions.glm(iPrediction).optThr = optThr;
        end
        % save groundtruth
        design.patients(iPatient).predictions.groundtruth = afxVolumeWrite(fullfile(patDestDir,[prefix 'groundtruth_masked.nii']),afxDeMask(masks.analysis,y(iPatient,:) & ~isnan(predictions.(predNames{1})(iPatient,:))),space.dim,'uint8',space.mat);
        % save predictors
        afxSaveVars(fullfile(patDestDir,[prefix 'predictors']),['intercept' design.predictors],[1 design.xRaw(iPatient,:)]);
        % save info
        afxSaveVars(fullfile(patDestDir,[prefix 'info']),{'fold' 'FWHM' 'minPerfusion' 'minLesion' 'optimalThreshold'},{design.fold design.FWHM design.minPerfusion design.minLesion optThr});
    end
    patients = design.patients;
end
