function [design] = afxEliminateFactor(design,factor)

    idxFactor = find(strcmpi(design.predictors,factor),1);
    if isempty(idxFactor)
    	error(['Unbekannter Praediktor "' factor '".']);
    end
    
    design.predictors(idxFactor) = [];
    
    for iPatient = 1:length(design.patients)
        design.patients(iPatient).xRaw(idxFactor) = [];
    end
end
