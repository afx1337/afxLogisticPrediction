function afxPlot2(x,y,labels,variant,colors,nbins,hd,a)
    %subMean = mean(y,2);
    %y = y-subMean+mean(subMean);
    %line([min(x)-1 max(x)+1],[0,0],'Color',[0.65 0.65 0.65],'LineStyle','--')    
    hold all
    if exist('colors','var')
        afxPlotData(x,y,nbins,hd,a,colors)
    end
    boxplot(y,'colors',[0 0 0],'boxstyle','outline','labelorientation','inline','outliersize',4000,'positions',x,'symbol','ko')
    %scatter(x,mean(y',2),'ok','filled')
    xticks(x);
    xticklabels(labels);
    xtickangle(60);
    title(strrep(variant,'_',' '));
    box('off')
    hold off
end



function afxPlotData(x,y,nbins,hd,a,colors)
    yl = [min(y(:)) max(y(:))];
    delta = (yl(2)-yl(1))/nbins;
    edges = yl(1):delta:yl(2);
    for group = 1:size(y,2)
        n = histcounts(y(:,group),edges);
        for i = 1:length(n)
            for j = 1:n(i)
                scatter(x(group)-n(i)*hd/2+(j-.5)*hd,edges(i)+delta/2,a,colors{group},'filled')
            end
        end
    end
    %set(findobj(gca,'type','line'),'linew',2);
    %set(gca,'linew',2)
    %ylim([yl(1)-diff(yl)/20 yl(2)+diff(yl)/20]);
    %xlim([x(1)-.5 x(2)+.5]);
end
