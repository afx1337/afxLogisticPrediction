function [x,design] = afxAddInteractions(x,design)
    % only works for two-way-interactions if at least on variable is encoded binary
    
    % https://stats.stackexchange.com/questions/11645/coding-an-interaction-between-a-nominal-and-a-continuous-predictor-for-logistic
     for iInteracton = 1:length(design.interactions)
        % find all predictors
        for iPredictor = 1:length(design.interactions(iInteracton).val)
            idxInteraction(iPredictor) = find(strcmpi(design.predictors,design.interactions(iInteracton).val{iPredictor}),1);
            if isempty(idxInteraction(iPredictor))
                error(['Interaktion mit unbekanntem Praediktor "' design.interactions(iInteracton).val{iPredictor} '".']);
            end
        end
        % calculate interaction
        x(:,end+1,:) = prod(x(:,idxInteraction,:),2);
        % save name of interaction
        design.predictors(end+1) = join(design.interactions(iInteracton).val,' x ');
        % updata xRaw
        for iPatient = 1:length(design.patients)
            design.patients(iPatient).xRaw{end+1} = 'interaction';
        end
    end
end
