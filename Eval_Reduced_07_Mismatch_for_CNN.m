clear all
clc

dEval.GLM = dir('data\*\output\reduced_model_07*-s9\evaluation\evalMismatch.mat');
dEval.Thr = dir('data\*\output\reduced_model_07*-s5\evaluation\evalMismatch.mat');

methods = { 'GLM' 'Thr' };
compartiments  = { 'core' 'penumbra' };
tici = { 'tici0' 'tici1' };

addpath('scripts');

for iGroup = 3:length(dEval.GLM)
	figure('units','normalized','outerposition',[0 0 .65 .65]);
    dat.design = load(fullfile(dEval.GLM(iGroup).folder,'..\design.mat'));
   	[~,groupName] = fileparts(dat.design.design.dataDir);
    if contains(dat.design.design.analysisName,'excl_tici_2a')
        groupName = strcat(groupName,'_excl_tici_2a');
    end
   	sgtitle(strrep(groupName,'_',' '));
    for iMethod = 1:length(methods)
        % load evaluation metrics and design
        dat.eval = load(fullfile(dEval.(methods{iMethod})(iGroup).folder,dEval.(methods{iMethod})(iGroup).name));
        %dat.design = load(fullfile(dEval.(methods{iMethod})(iGroup).folder,'..\design.mat'));
        ticiScore = [dat.eval.tbl.(strcat('pred',num2str(find(strcmp(dat.eval.predictors,'tici')))))] > .5;
        for iCompartiment = 1:length(compartiments)
            tmp_infarctionRate = [dat.eval.tbl.([methods{iMethod} '_' compartiments{iCompartiment}])]';
            tmp_patID = {dat.eval.tbl.name}';
            tmp_patID = cell2mat(cellfun(@str2num,tmp_patID(:),'un',0));
            for iTici = 1:length(tici)
                r.(methods{iMethod}).(tici{iTici}).(compartiments{iCompartiment}).infarctionRate = tmp_infarctionRate(ticiScore == (iTici-1));
                r.(methods{iMethod}).(tici{iTici}).(compartiments{iCompartiment}).patientID = tmp_patID(ticiScore == (iTici-1));
            end
        end
        % plot mismatch
        subplot(1,2,iMethod);
        y1 = [ r.(methods{iMethod}).tici0.core.infarctionRate r.(methods{iMethod}).tici0.penumbra.infarctionRate ];
        y2 = [ r.(methods{iMethod}).tici1.core.infarctionRate r.(methods{iMethod}).tici1.penumbra.infarctionRate ];
        y = nan(max(length(y1),length(y2)),4);
        y(1:length(y1),1:2) = y1;
        y(1:length(y2),3:4) = y2;
        labels = {'TICI < 2b x core' 'TICI < 2b x penumbra' 'TICI >= 2b x core' 'TICI >= 2b x penumbra'};
        colors = { [0,92,171]/255, [220,238,243]/355,  [0,92,171]/255, [220,238,243]/355 };
        afxPlot2([1 2 3.5 4.5],y,labels,methods{iMethod},colors,80,0.02,15);
        ylim([-.04 1.04])
    end
    % save figure
    destDir = fullfile('data','evaluation_reduced07_mismatch_CNN');
    mkdir(destDir);
    print(gcf,fullfile(destDir,['Mismatch_' groupName '.png']),'-dpng','-r120');
    save(fullfile(destDir,['Mismatch_' groupName '.mat']),'r')
end


rmpath('scripts');