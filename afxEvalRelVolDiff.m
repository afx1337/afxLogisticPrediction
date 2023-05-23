function relDiff = afxEvalRelVolDiff(a,b)
    % b ... truth
    % a ... prediction
    
    relDiff = nan(1,size(a,1));
	parfor iPatient = 1:size(a,1)
        relDiff(iPatient) = abs(nnz(a(iPatient,:))-nnz(b(iPatient,:)))/nnz(b(iPatient,:));
        if isinf(relDiff(iPatient)), relDiff(iPatient) = NaN; end
	end
end