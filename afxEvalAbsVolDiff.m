function absDiff = afxEvalAbsVolDiff(a,b)
    
    absDiff = nan(1,size(a,1));
    parfor iPatient = 1:size(a,1)
        absDiff(iPatient) = abs(nnz(a(iPatient,:))-nnz(b(iPatient,:)));
    end
end
