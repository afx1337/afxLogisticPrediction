function afxWriteVars(fname,names,vals)
    fileID = fopen([fname '.txt'],'w');
    for i = 1:length(names)
        val = vals{i};
        info(i).name = names{i};
        info(i).value = val;
        if isnumeric(val), val = num2str(val); end
        if strcmp(val,'')
            fprintf('\n')
        else
            fprintf(fileID,'% 20s:    %s\n',names{i},val);
        end
    end
    fclose(fileID);
    save([fname '.mat'],'info');
end
