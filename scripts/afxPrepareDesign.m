function [x,y,masks,design] = afxPrepareDesign(design,space,brainmask,idxMask)

    if ~exist('idxMask','var')
        idxMask = true(1,length(design.patients)); 
    end

    nPatients = length(design.patients);
    nPredictors = length(design.predictors);
    nVoxels = prod(space.dim);

    % load images into design matrix x and response y
    s = tic;
    fprintf('Loading images ... ');
    x = nan(nPatients,nPredictors,nVoxels);
    y = false(nPatients,nVoxels);    %%creates matrix of the dimension npatients x nVoxels
    for iPatient = 1:nPatients    
        for iPredictor = 1:nPredictors
            val = design.patients(iPatient).xRaw{iPredictor};
            if isnumeric(val)
                x(iPatient,iPredictor,:) = val;
            else
                x(iPatient,iPredictor,:) = afxVolumeResample(val,space.XYZmm,1);
            end
        end
        y(iPatient,:) = afxVolumeResample(design.patients(iPatient).yRaw,space.XYZmm,0) > .5;
        % if a patient has a old lesion, remove from x and y
        if ~isempty(design.patients(iPatient).yRawOld)
            ytmp = afxVolumeResample(design.patients(iPatient).yRawOld,space.XYZmm,0) > .5;
            y(iPatient,ytmp) = 0;
            x(iPatient,:,ytmp) = NaN;
        end
    end
    fprintf('done (%.2f min).\n',toc(s)/60);

    s = tic;
    fprintf('Preparing masks and smoothing data ... ');
    % find cbf
    idxCBF = find(strcmpi(design.predictors,'cbf'),1);
    % make CBF implizit mask explizit
    xZero = x == 0;
    xCBF = false(size(x)); xCBF(:,idxCBF,:) = 1;
    x(xZero & xCBF) = NaN;
    % mask lesion with individual perfusion mask
    y = squeeze(~isnan(x(:,idxCBF,:))).*y;
    % create masks
    masks.perfusion = squeeze(sum(~isnan(x(idxMask,idxCBF,:))))';
    masks.lesions = sum(y(idxMask,:));
    masks.analysis = logical(round(afxVolumeResample(brainmask,space.XYZmm,1)'));
    masks.informative = (masks.perfusion > design.minPerfusion*(nPredictors+1+length(design.interactions))) & (masks.lesions > nPatients*design.minLesion);
    % replace NaNs with 0 before smoothing
    xNaN = isnan(x); x(xNaN) = 0;
    % avoid cbf thesholding edge artifacts with low values due to smoothing
    x(xNaN & xCBF) = 30; % (100 = group mean in unaffected hemisphere)
    % smoothing
    for j = find(cellfun(@isstr,design.patients(1).xRaw))
        x(:,j,:) = afxFastSmooth(x(:,j,:),design.FWHM,space.dim,space.mat);
    end
    % restore explicit masks
    x(xNaN) = NaN;
    % discard data outside mask
    y = y(:,masks.analysis);      
    x = x(:,:,masks.analysis);  
    % add interactions
    [x,design] = afxAddInteractions(x,design);
    fprintf('done (%.2f min).\n',toc(s)/60);
end
