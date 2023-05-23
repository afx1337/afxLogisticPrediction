function y = afxVolumeResample(volume,XYZorig,hold)
    % dat = afxVolumeResample(fname,XYZ,hold)
    % resample fname to space defined by XYZ and return data as (linearly
    % packed) vector
    % hold refers to interpolation method for the resampling:
    %       0         : Zero-order hold (nearest neighbour)
    %       1         : First-order hold (trilinear interpolation)
    %       2->127    : Higher order Lagrange (polynomial) interpolation
    %                   using different holds (second-order upwards)
    %       -127 - -1 : Different orders of sinc interpolation
    
    if nargin < 3, hold = 0; end
    % load volume
    Vresample = spm_vol(volume);
    % compute voxel coordinates in voxel space of volume wich correspond to
    % the space given by XYZ
    RCPresample = Vresample.mat\XYZorig;
    % resample volume
    y  = spm_sample_vol(Vresample,RCPresample(1,:),RCPresample(2,:),RCPresample(3,:),hold);
    % pack to vector
    y = y(:);
end