% TASK 1
%{
clear           % clear all variables
close all       % close all figure windows
imfinfo('assets/breastXray.tif')
f = imread('assets/breastXray.tif');
imshow(f)

intensityPixel_3_10 = f(3,10)             % print the intensity of pixel(3,10)
imshow(f(1:286,:))  % display only top half of the image - (571/2 = 285.5 ~= 286)

[fmin, fmax] = bounds(f(:)) % get minimum and maximum values of intensity within the whole image
% we get 21 to 255 -- close to the full range [0,255]

figure  % open a new figure window
imshow(f(:,241:482))  % display only right half of the image

g1 = imadjust(f, [0 1], [1 0]); % invert image: remaps the values in the intensity range, f, from min,max to max,min. 
figure                          % open a new figure window
imshowpair(f, g1, 'montage', 'Scaling','none') % we use 'Scaling','none' to avoid the visuals from being normalized.

g2 = imadjust(f, [0.5 0.75], [0 1]); % Remaps values withing the 0.5-0.75 range to the extremes (0,1). 
g3 = imadjust(f, [ ], [ ], 2); % Applies gamma correction with value 2. This achieves similar results but preserves more details because g2 simply truncated the histogram. 
figure
montage({g2,g3})
%}

% TASK 2

%{
clear all       % clear all variables
close all       % close all figure windows
f = imread('assets/bonescan-front.tif');
r = double(f);  % uint8 to double conversion
k = mean2(r);   % find mean intensity of image - mean2 function computes mean in a 2D matrix.
E = 0.9;        % stepness of the contrast stretching function
s = 1 ./ (1.0 + (k ./ (r + eps)) .^ E); % r could be zero! That's why we add eps. eps is a Matlab variable for a very small value that avoids division by 0.
g = uint8(255*s);
imshowpair(f, g, "montage")

%}

% TASK 3

%{
clear all       % clear all variable in workspace
close all       % close all figure windows
f=imread('assets/pollen.tif');
imshow(f)
figure          % open a new figure window
imhist(f);      % calculate and plot the histogram

close all
g=imadjust(f,[0.3 0.55]); % remap and stretch histogram in the 0.3-0.55 range.
montage({f, g})     % display list of images side-by-side
figure
imhist(g);

% Let's normalize the histogram
close all                       
g_pdf = imhist(g) ./ numel(g);  % compute PDF - probability distribution function for the intensity function of the already remapped function g.
g_cdf = cumsum(g_pdf);          % compute CDF - cumulative sum of the probability distribution function
imshow(g);
subplot(1,2,1)                  % Makes space for plot 1 in a 1x2 subplot
plot(g_pdf)                     % Plots the PDF
subplot(1,2,2)                  % Makes space for plot 2 in a 1x2 subplot
plot(g_cdf)                     % Plots the CDF

% The CDF is the function used as the intensity transformation function.

x = linspace(0, 1, 256);    % x has 256 values equally spaced between 0 and 1
figure
plot(x, g_cdf)              % Showing the remapping function
axis([0 1 0 1])             % graph x and y range is 0 to 1 (0: minimum intensity, 1: maximum intensity)
set(gca, 'xtick', 0:0.2:1)  % x tick marks are in steps of 0.2
set(gca, 'ytick', 0:0.2:1)
xlabel('Input intensity values', 'fontsize', 9)
ylabel('Output intensity values', 'fontsize', 9)
title('Intensity transformation function', 'fontsize', 12)

% To equalize the histogram using MATLAB's inbuilt function
h = histeq(g,256);              % MATLAB function for histogram equalize g
% Or alternatively, we can do it ourselves using the CDF:
g_normalized = g_cdf(double(g) + 1);  % Use double for pixel indexing
g_equalized = uint8(255 * g_normalized); 

close all
montage({f, g, h, g_equalized})              % Showing all 3 images compared (original; remapping 0.3-0.55 to 0.0-1.0, and histogram normalization).
figure;
subplot(1,3,1); imhist(f);
subplot(1,3,2); imhist(g);
subplot(1,3,3); imhist(h);
%}


% TASK 4:


clear all
close all
f = imread('assets/noisyPCB.jpg');
imshow(f)

w_box = fspecial('average', [9 9]); % Box filter. 9x9 kernel. 
w_gauss = fspecial('Gaussian', [7 7], 0.5); % Gaussian filter, with sigma = 0.5. Note how the coefficients are adding to 1.0 so we preserve intensity.

g_box = imfilter(f, w_box, 0);
g_gauss = imfilter(f, w_gauss, 0);
figure
montage({f, g_box, g_gauss})

% Thoughts: Low pass filters always reduce noise, but at the cost of losing
% sharpness. In this regard, the box filter is worse, because it does not
% assign any weight relative to the radial distance of the neighborhood pixel 
% from the pixel being analyzed -- and instead it just averages everything
% out. Also, the kernel size is bigger (9x9 > 7x7), making for a more blurry result.
% The Gaussian filter is more appropriate, because it smoothens intensity
% changes better by assigning greater weights to the pixels that are closer
% to the pixel in question. 

g_median = medfilt2(f, [7 7], 'zero');
figure; montage({f, g_median})

% Thoughts: the median filter is the kernel is an interesting filter. It has
% no coefficients. We overlay it pixel by pixel, just like before.
% And then we observe the pixels in that mask overlay, and arrange 
% them by value. Then we take the middle one. In other words: we find the pixel
% with the median value in that neighborhood, and we output that value! 