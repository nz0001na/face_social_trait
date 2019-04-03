% By Wang Shuo Jan 20 2010
% Modified by Wang Shuo Apr 16 2012
% should use sum instead of length for ~isnan(x)

function plotBar(x,barPos,barColor)

mx = nanmean(x);
% mx = nanmedian(x);

stdx = nanstd(x);
sex = stdx / sqrt(sum(~isnan(x)));

barL = 0.3;
bar(barPos,mx,barColor);
plot([barPos barPos],mx+sex*[-1 1],'k')
plot(barPos+[barL -barL],mx+sex*[1 1],'k')
plot(barPos+[barL -barL],mx-sex*[1 1],'k')