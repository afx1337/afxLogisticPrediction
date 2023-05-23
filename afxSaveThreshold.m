function [patients] = afxSaveThreshold(thresholdMaps,masks,space,design)
    % thresholdMaps(1).dat = data(nPat,nVox);
    % thresholdMaps(1).name = 'CBFrel';
    % thresholdMaps(1).inverse = true;
    % thresholdMaps(1).optThr = .3;
    % thresholdMaps(2).dat = data(nPat,nVox);
    % thresholdMaps(2).name = 'Tmax';
    % thresholdMaps(2).inverse = false;
    % thresholdMaps(2).optThr = 6;
    
    % file prefix
    prefix = 'Threshold_';

    % destination directory
    destDir = fullfile(design.dataDir,'output',strcat(design.analysisName,'-s',num2str(design.FWHM)),'predictions');
    
    for iPatient = 1:length(design.patients)
        % patient dir
        patDestDir = fullfile(destDir,design.patients(iPatient).name,'threshold');
        mkdir(patDestDir);
        % save cbf/tmax
        for iThrMap = 1:length(thresholdMaps)
            fname = sprintf('%smap_%s.nii',prefix,thresholdMaps(iThrMap).name);
            design.patients(iPatient).predictions.thr(iThrMap).name = thresholdMaps(iThrMap).name;
            design.patients(iPatient).predictions.thr(iThrMap).file = afxVolumeWrite(fullfile(patDestDir,fname),afxDeMask(masks.analysis,thresholdMaps(iThrMap).dat(iPatient,:)),space.dim,'int16',space.mat);
            design.patients(iPatient).predictions.thr(iThrMap).optThr = thresholdMaps(iThrMap).optThr;
            design.patients(iPatient).predictions.thr(iThrMap).inverse = thresholdMaps(iThrMap).inverse;
        end
        % save info
        optThrNames = strcat('optThr_',{ thresholdMaps.name });
        optThrVals = { thresholdMaps.optThr };
        afxWriteVars(fullfile(patDestDir,[prefix 'info']),[{'fold' 'FWHM' 'minPerfusion' 'minLesion'} optThrNames],[{design.fold design.FWHM design.minPerfusion design.minLesion} optThrVals]);
    end
    patients = design.patients;
end
