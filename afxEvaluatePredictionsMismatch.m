function [] = afxEvaluatePredictionsMismatch(designFile)

    % get design file via ui if not provided
    if ~exist('designFile','var')
        [~,pwdD] = fileparts(pwd);
        if strcmp(pwdD,'scripts'), cd('..'); end
        if ~exist('designFile','var')
            [designF,designP] = uigetfile('data\design.mat','Select design ...');
            designFile = fullfile(designP,designF);
        end
    end
    
    % laod design file
    load(designFile,'design');
    
    % find tici predictor
    idxTici = strcmpi(design.predictors,'tici');
    
    % load analysis mask
    [dat.mask,space.XYZmm,space.dim,space.mat] = afxVolumeLoad(fullfile(design.dataDir,'output',strcat(design.analysisName,'-s',num2str(design.FWHM)),'models','fold001','LogisticGLM_mask_analysis.nii'));
    dat.mask = dat.mask > .5;
    
    % for every patient ...
    pct = length(design.patients)/50;
    s = tic;
    fprintf('Calcul. eval. metrics [');
    for iPatient = 1:length(design.patients)
        % load groundtruth
        fileGroundtruth = design.patients(iPatient).predictions.groundtruth;
        [dat.gt] = afxVolumeLoad(fileGroundtruth);
        dat.gt = dat.gt > .5;
        % get tici score
        tici = design.patients(iPatient).xRaw{idxTici};
        % get prediction maps -> pred
        pred.GLM.tici1 = design.patients(iPatient).predictions.glm(2);
        pred.Thr.tici1 = design.patients(iPatient).predictions.thr(1);
        pred.GLM.tici0 = design.patients(iPatient).predictions.glm(1);
        pred.Thr.tici0 = design.patients(iPatient).predictions.thr(2);
        % write name and predictors to table
        tbl(iPatient).name = design.patients(iPatient).name;
        for iPred = 1:length(design.predictors)
            tbl(iPatient).(['pred',num2str(iPred)]) = design.patients(iPatient).xRaw{iPred};
        end
        %  vertical seperator
        tbl(iPatient).vsep1 = ' ';
        tbl(iPatient).sizeGroundtruth = nnz(dat.gt)*prod(sqrt(sum(space.mat(1:3,1:3).^2)))/1000; % ml
        tbl(iPatient).vsep2 = ' ';
        % write evaluation metrics to table
        modalities = fieldnames(pred);
        for iModality = 1:length(modalities)
            predictions = fieldnames(pred.(modalities{iModality}));
            for iPrediction = 1:length(predictions)
                % load prediction
                [dat.pred.(predictions{iPrediction})] = afxVolumeLoad(pred.(modalities{iModality}).(predictions{iPrediction}).file);
                % binarize prediction
                if pred.(modalities{iModality}).(predictions{iPrediction}).inverse
                    dat.pred.(predictions{iPrediction}) = dat.pred.(predictions{iPrediction}) < pred.(modalities{iModality}).(predictions{iPrediction}).optThr & dat.pred.(predictions{iPrediction}) > 0;
                else
                    dat.pred.(predictions{iPrediction}) = dat.pred.(predictions{iPrediction}) > pred.(modalities{iModality}).(predictions{iPrediction}).optThr;
                end
            end
            % calculate core and penumbra
            dat.pred.core = dat.pred.tici1;
            dat.pred.penumbra = dat.pred.tici0 & ~dat.pred.tici1;
            
            % evaluation metrics based on binarized prediction
            if tbl(iPatient).sizeGroundtruth == 0
                tbl(iPatient).(strcat(modalities{iModality},'_core')) = NaN;
                tbl(iPatient).(strcat(modalities{iModality},'_penumbra')) = NaN;
            else
                tbl(iPatient).(strcat(modalities{iModality},'_core')) = mean(dat.gt(dat.pred.core));
                tbl(iPatient).(strcat(modalities{iModality},'_penumbra')) = mean(dat.gt(dat.pred.penumbra));
            end
            %  vertical seperator
            if iModality < length(modalities), tbl(iPatient).(strcat('vsep',num2str(iModality+2))) = ' '; end
        end
        if mod(iPatient,pct) < 1, fprintf('.'); end
    end
    fprintf('] (%.2f min)\n',toc(s)/60);
    
    % save to disk
	outDir = fullfile(design.dataDir,'output',strcat(design.analysisName,'-s',num2str(design.FWHM)),'evaluation');
    mkdir(outDir);
    tblCell = [fieldnames(tbl)';struct2cell(tbl')'];
    tblCell(1,2:1+length(design.predictors)) = design.predictors;
    xlswrite(fullfile(outDir,'evalMismatch.xls'),tblCell);
    predictors = design.predictors;
    save(fullfile(outDir,'evalMismatch.mat'),'tbl','predictors');
end