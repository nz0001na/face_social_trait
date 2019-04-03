% By Shuo Wang Jan 19 2019
% combine frames into video

clear;

imgDir = 'J:/1_keypoint_observe_8/1_png_scenario8_visual_rename/Alex_150727_2_shorten_with_text_v3';

imgNames = dir([imgDir filesep '*.png']);

vidObj = VideoWriter('Alex_150727_2_v3.avi');
open(vidObj);

for iFrame = 1:length(imgNames)
    img = imread([imgDir filesep imgNames(iFrame).name]);
    writeVideo(vidObj,img);
end

close(vidObj);