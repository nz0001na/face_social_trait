clear
clc
close all

myimage = imread('frame_000001.png');
myimage = im2double(myimage);

% imshow(myimage);

[r,c,~] = size(myimage);

set(gca,'units','pixels'); %// set the axes units to pixels
x = get(gca,'position'); %// get the position of the axes
set(gcf,'units','pixels'); %// set the figure units to pixels
y = get(gcf,'position'); %// get the figure position
set(gcf,'position',[y(1) y(2) x(3) x(4)]);% set the position of the figure to the length and width of the axes
set(gca,'units','normalized','position',[0 0 1 1]) % set the axes units to pixels

%// Example
hold on
mynumber = '6777';

%// Set up position/size of the box
Box_x = 25;
Box_y = 25;
Box_size = [300 300];
%// Beware! For y-position you want to start at the top left, i.e. r -
%// Box_y
BoxPos = [Box_x r-Box_y Box_size(1) Box_size(2)];
MyBox = uicontrol('style','text','Position',BoxPos);

set(MyBox,'String',mynumber);

% figure
imshow(myimage)
saveas(myimage,'1.jpg')
% saveas(myimage,'1.png')
