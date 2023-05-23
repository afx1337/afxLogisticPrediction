function meanDist = afxEvalDist(a,b,space)
    % b ... truth
    % a ... prediction

    meanDist = nan(1,size(a,1));
    parfor iPatient = 1:size(a,1)
        % https://en.wikipedia.org/wiki/Hausdorff_distance
        img1 = a(iPatient,:) > .5;
        img2 = b(iPatient,:) > .5;
        d = afxDist(img1,img2,space);
        meanDist(iPatient) = mean(d);
    end
end

function d = afxDist(img1,img2,space)
    d = zeros(1,length(img1));
    overlap = img1 & img2;
    img1(overlap) = false;
    [img2, ~] = afxExtrSurface(img2,space.dim);
    xyz1 = space.XYZmm(1:3,img1);
    xyz2 = space.XYZmm(1:3,img2);
    if isempty(xyz1) || isempty(xyz2)
        d = NaN;
        return
    end
    len1 = size(xyz1,2);
    for i = 1:len1
        d(i) = sqrt(min(sum((xyz2-xyz1(i)).^2,1)));
    end
end

function [datSurface, datCore] = afxExtrSurface(datLesion,dim)
    datLesion = reshape(datLesion,dim);
    siz = size(datLesion);
    lesionInd = find(datLesion);
    datSurface = false(siz);
    for i = 1:length(lesionInd)
        ind = lesionInd(i);
        [x,y,z] = ind2sub(siz,ind);
        if x == 1 || y == 1 || z == 1 || x == siz(1) || y == siz(2) || z == siz(3)
            datSurface(x,y,z) = true;
        else
            datSurface(x,y,z) = (~datLesion(x-1,y,z) | ~datLesion(x+1,y,z) | ~datLesion(x,y-1,z) | ~datLesion(x,y+1,z) | ~datLesion(x,y,z-1) | ~datLesion(x,y,z+1));
        end
    end
    datSurface = datSurface(:);
    datCore = datLesion(:) & ~datSurface;
end