function y = afxFastSmooth(y,FWHM,dim,mat)
    % y = afxSmoooth(y,FWHM,dim,mat)
    % convolve data in y (linearly packed in the spatial dimension with
    % original dimensions dim and q-form matrix mat) with gausian smoothing
    % kernel with full width at half maximum of FWHM

    if length(FWHM) == 1, FWHM = [FWHM FWHM FWHM]; end
    % check if image toolbox is installed
    if (exist('imgaussfilt3','file') == 2)
        spm = false;
        % adapt fwhm to voxel size and calculate sigma
        sigma = FWHM./sqrt(8*log(2))./sqrt(sum(mat(1:3,1:3).^2));
    else
        spm = true;
        % adapt fwhm to voxel size
        FWHM = FWHM./sqrt(sum(mat(1:3,1:3).^2));
    end
    
    if FWHM ~= [0 0 0]
        % generate dummy matrix for smoothed data
        if spm, sdat = nan(dim); end
        % iterate over all images
        for i = 1:size(y,1)
            % perform smoothing
            if spm
                spm_smooth(reshape(y(i,:),dim),sdat,FWHM);
            else
                sdat = imgaussfilt3(reshape(y(i,:),dim),sigma,'padding',0,'FilterDomain','spatial');
            end
            % copy data back to y
            y(i,:) = sdat(:);
        end
    end
end