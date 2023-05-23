function [y,XYZmm,dim,mat] = afxVolumeLoad(fnameNii)
    % [y,XYZmm,dim,mat] = afxVolumeLoad(fnameNii)
    
    % load nifti using spm functions
    Vfunc = spm_vol(fnameNii);
    [y,XYZmm] = spm_read_vols(Vfunc);
    % vectorize data
    y = y(:);
    % fourth dimension of world space
    XYZmm = [XYZmm; ones(1,size(XYZmm,2))];
    % copy variables
    dim = Vfunc.dim;
    mat = Vfunc.mat;
end