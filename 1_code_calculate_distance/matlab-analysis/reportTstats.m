% By Shuo Wang Jun 8 2015
% report t-test stats

function reportTstats(A,B,paired,nameA,nameB)

if nargin < 3
    paired = 0;
end

if nargin < 4
    nameA = 'A';
    nameB = 'B';
end

if paired
    [h p c d] = ttest(A,B);
else
    [h p c d] = ttest2(A,B);
end

% g = (c(1)+c(2))/2/d.sd;

% if isempty(which('mes'))
%     addpath(genpath('./EffectSizeToolbox_v1.3/EffectSizeToolbox_v1.3/EffectSizeToolbox_v1.3/'))
% end

tmp = mes(A',B','hedgesg');
g = tmp.hedgesg;

disp(['Mean±SD ' nameA ': ' num2str(nanmean(A),3) '  ' num2str(nanstd(A),3)]);
disp(['Mean±SD ' nameB ': ' num2str(nanmean(B),3) '  ' num2str(nanstd(B),3)]);

disp(['t(' num2str(d.df) ') = ' num2str(d.tstat,3)]);
disp(['P-val = ' num2str(p,3)]);
disp(['Hedges g = ' num2str(g,3)]);

% if exist('permutationTest')
    pp = permutationTest(A,B);
    disp(['Permutation P=' num2str(pp,3)]);
end


