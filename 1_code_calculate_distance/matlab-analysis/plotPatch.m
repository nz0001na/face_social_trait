% By Wang Shuo Feb 24 2012
% plot patch
% modified on Nov 21 2013
% compute ~isnan at each position

function plotPatch(x,M,c,isTruncate)

if nargin<4
    isTruncate = 0;
end

meanM = nanmean(M,1);
n = sum(~isnan(M));
seM = nanstd(M,[],1) ./ sqrt(n);

if isTruncate
    tmp = find(~isnan(meanM),1,'last');
    x = x(1:tmp);
    meanM = meanM(1:tmp);
    seM = seM(1:tmp);
end

patch([x x(end:-1:1)], [meanM+seM meanM(end:-1:1)-seM(end:-1:1)],c)