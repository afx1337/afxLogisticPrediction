function mcc = afxEvalMCC(a,b)
    
    mcc = nan(1,size(a,1));
    parfor iPatient = 1:size(a,1)
        % https://en.wikipedia.org/wiki/Phi_coefficient
        tp = nnz(a(iPatient,:)&b(iPatient,:));
        tn = nnz(~a(iPatient,:)&~b(iPatient,:));
        fp = nnz(a(iPatient,:)&~b(iPatient,:));
        fn = nnz(~a(iPatient,:)&b(iPatient,:));
        mcc(iPatient) = (tp*tn-fp*fn) / sqrt( (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) );
    end
end