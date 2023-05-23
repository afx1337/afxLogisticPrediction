function AUC = afxEvalROCAUC(scores,labels,render)
    if ~exist('render','var')
        render = false;
    end
    
    AUC = nan(1,size(labels,1));
    if render, figure(); hold all; end
	for iPatient = 1:size(labels,1)
        if nnz(labels(iPatient,:)) == 0 || nnz(~isnan(scores(iPatient,labels(iPatient,:)))) == 0
            AUC(iPatient) = NaN;
        else
            [x,y,~,AUC(iPatient)] = perfcurve(labels(iPatient,:),scores(iPatient,:),true);
            if render, plot(x,y); end
        end
    end
    if render
        xlabel('False positive rate');
        ylabel('True positive rate');
        title('ROC for Classification by Logistic Regression');
    end
end