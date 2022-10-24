% MATAB function which takes an image and outputs the coordinates of the
% boundary of that image. 
function [c1_prime,c2_prime]=image_processing(path_to_file,white_background,filter_rate,do_plot,thresh)

I= imread(path_to_file); % read image 

[~, ~, numberOfColorChannels] = size(I); % check if image is color
if numberOfColorChannels > 1
    I_gray = rgb2gray(I); % if color, then convert to grey
else
    I_gray=I;
end

if nargin==5 % check if threshold value for grey -> binary is given
    IB=im2bw(I,thresh);
else
    IB=im2bw(I,mean(I_gray,'all')/255); % if not, then use the 
    % average image brightness to guess this values
    %( Generally, this doesn't work well so I'd reccomend trial and error
    %to get this value)
end

% if image has a white background then we need to swap the black and white
% as MATLAB functions are desgined to look for bright objects on dark
% backgrounds
if white_background
    IB=~IB;
end

[labeledImage,~]=bwlabel(IB, 8);  % fine objects in image
props=regionprops(labeledImage, I_gray, 'all');  % get properties of objects
allBlobAreas = [props.Area];  % get vector of object areas
mesh_indices=allBlobAreas==max(allBlobAreas);  % we assume we want the largest area object
mesh_image=ismember(labeledImage, find(mesh_indices)); 
boundaries=bwboundaries(mesh_image);  % get boundaries of this object
b=boundaries{1};

c1  = downsample(b(:,1),filter_rate); % downsample image if filter_rate>1, 
% this results in faster mesh generation as well as avoiding 
% self-intersections sometimes
c2  = downsample(b(:,2),filter_rate);

% normalise image such that the laregest dimension sits [-0.5,0.5]
c1_prime=c1-mean(c1);
c2_prime=c2-mean(c2);
x_length=max(c2_prime)-min(c2_prime);
y_length=max(c1_prime)-min(c1_prime);
characteristic_size=max([x_length,y_length]);
c1_prime=c1_prime./characteristic_size;
c2_prime=c2_prime./characteristic_size;

% show image with the boundary we found if asked
if do_plot
    figure;
    imshow(I_gray)
    hold on
    plot(c2,c1)
end