clear
clc
close all

speed_path = 'J:/1_keypoint_observe_8/1_png_scenario8_visual_rename/Alex_150727_2_speed/'
left_data = csvread([speed_path 'left_speed.csv']); 
right_data = csvread([speed_path 'right_speed.csv']); 

des_path = 'J:/1_keypoint_observe_8/1_png_scenario8_visual_rename/Alex_150727_2_shorten_with_text/'

imgDir = 'J:/1_keypoint_observe_8/1_png_scenario8_visual_rename/Alex_150727_2_shorten_rename';
imgNames = dir([imgDir filesep '*.png']);

for iFrame = 2:length(imgNames)
    name = imgNames(iFrame).name
    rename = name(1:end-4)
    myimage = imread([imgDir '/' name]);
    myimage = im2double(myimage);

    imshow(myimage);

    [r,c,~] = size(myimage);

    set(gca,'units','pixels'); %// set the axes units to pixels
    x = get(gca,'position'); %// get the position of the axes
    set(gcf,'units','pixels'); %// set the figure units to pixels
    y = get(gcf,'position'); %// get the figure position
    set(gcf,'position',[y(1) y(2) x(3) x(4)]);% set the position of the figure to the length and width of the axes
    set(gca,'units','normalized','position',[0 0 1 1]) % set the axes units to pixels

    
    index = int8((iFrame + 1)/30 +1 )
    left = left_data(index)
    right = right_data(index)
    
    hold on
    mynumber = ['left:' num2str(left) ' ; right:' num2str(right)];

    %// Set up position/size of the box
    Box_x = 25;
    Box_y = 25;
    Box_size = [300 15];
    %// Beware! For y-position you want to start at the top left, i.e. r -
    %// Box_y
    BoxPos = [Box_x r-Box_y Box_size(1) Box_size(2)];
    MyBox = uicontrol('style','text','Position',BoxPos);

    set(MyBox,'String',mynumber);

    % figure
    imshow(myimage)
    savefig([des_path rename '.fig'])
    
    
%     img = imread([imgDir filesep imgNames(iFrame).name]);
%     writeVideo(vidObj,img);
end




