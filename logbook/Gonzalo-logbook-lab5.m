%% TASK 1: POINT DETECTION

%{
clear all
close all
f = imread('assets/crabpulsar.tif');
w = [-1 -1 -1; % Laplacian kernel to accentuate and detect isolated points.
     -1  8 -1;
     -1 -1 -1]; 
g1 = abs(imfilter(f, w));     % point detected
se = strel("disk",1); % structural element of morphological filter with a disk shape and radius 1
g2 = imerode(g1, se);         % erode using morphological filter
threshold = 100;
g3 = uint8((g2 >= threshold)*255); % threshold to only detect point.
montage({f, g1, g2, g3});

%}

%% TASK 2: EDGE DETECTION

%{

clear all
close all
f1 = imread('assets/circles.tif');
f2 = imread('assets/brain_tumor.jpg');

[g, t] = edge(f1, 'sobel'); % Out of the box function to detect edges in MATLAB

%{
f is the input image.
'method' is one of several algorithm to be used for edge detection: Sobel ('sobel'),
LoG ('log'), Canny ('canny') 

g is the output image. 
t is an optional return value giving the threshold being used in the algorithm to produce the output.
%}

% f1
% Trying out different methods and thresholds
% a) Sobel
[sobelAuto, t] = edge(f1, 'sobel'); % Automatic threshold chosen 
[sobel0, t] = edge(f1, 'sobel', 0); % Zero threshold - overfitting: too sensitive
[sobel0_07, t] = edge(f1, 'sobel', 0.07); % 0.07 threshold - about right
[sobel0_5, t] = edge(f1, 'sobel', 0.5); % 0.1 threshold - underfitting: too insensitive

% montage({f1, sobelAuto, sobel0, sobel0_07, sobel0_5});

% b) Laplacian of Gaussian
[logAuto, t] = edge(f1, 'log'); % Automatic threshold chosen 
[log0_006, t] = edge(f1, 'log', 0.006); % 0.006 threshold - about right

% montage({f1, logAuto, log0_006});

% c) Canny edge detection

[cannyAuto, t] = edge(f1, 'canny'); % Automatic threshold chosen 
[canny0_15, t] = edge(f1, 'canny', 0.15); % 0.15 threshold - about right

% montage({f1, cannyAuto, canny0_15});

% Overall comparison of methods for f1
% montage({f1, sobelAuto, log0_006, canny0_15});

% f2
% Trying out different methods and thresholds
% a) Sobel
[sobelAuto, t] = edge(f2, 'sobel'); % Automatic threshold chosen 
[sobel0, t] = edge(f2, 'sobel', 0); % Zero threshold - overfitting: too sensitive
[sobel0_06, t] = edge(f2, 'sobel', 0.06); % 0.06 threshold - about right
[sobel0_5, t] = edge(f2, 'sobel', 0.5); % 0.1 threshold - underfitting: too insensitive

% montage({f2, sobelAuto, sobel0, sobel0_06, sobel0_5});

% b) Laplacian of Gaussian
[logAuto, t] = edge(f2, 'log'); % Automatic threshold chosen 
[log0_004, t] = edge(f2, 'log', 0.004); % 0.004 threshold - about right

% montage({f2, logAuto, log0_004});

% c) Canny edge detection

[cannyAuto, t] = edge(f2, 'canny'); % Automatic threshold chosen 
[canny0_15, t] = edge(f2, 'canny', 0.15); % 0.15 threshold - about right

% montage({f2, cannyAuto, canny0_15});

% Overall comparison of methods for f2
% montage({f2, sobelAuto, log0_004, canny0_15});

% In both instances, the canny edge detection allows for very well-defined
% edges. This is thanks to the non-maximum supression which shrinks the
% edges in the normal direction of the edge. 

%}


%% TASK 3: HOUGH DETECTION FOR LINE DETECTION

%{

% Remember: hough remaps into a parameter plane in which lines become
% points, and points become sinusoidal waves (we could remap into any other parametric 
% function that we may find useful, like a circle, but that is not strictly Hough transform).

% Why do we not remap into line equations of type mx + c? Because m could
% very easily equal infinity and we would run into many issues very easily.

% Read image and find edge points
clear all; close all;
f = imread('assets/circuit_rotated.tif');
fEdge = edge(f,'Canny'); % We use the canny method to find well-defined edge points. 
figure(1)
montage({f,fEdge})

% Apply hough transform - 

[H, theta, rho] = hough(fEdge);

% Display transform result: we can see certain points have higer intensity
% (peaks) -- these high density points can be remapped back into the 
% detected lines.

figure(2)
imshow(H,[],'XData',theta,'YData', rho, ...
            'InitialMagnification','fit');
xlabel('theta'), ylabel('rho');
axis on, axis normal, hold on;

% Find those peaks: 
figure(2)
peaks  = houghpeaks(H,5); % 5 largest peaks 
% We could also use a threshold as a condition for the peaks to be found. 
%peaks  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
x = theta(peaks(:,2)); y = rho(peaks(:,1));
plot(x,y,'o','color','red', 'MarkerSize',10, 'LineWidth',1);

% Plot the Hough image as a 3D plot (called SURF) - this way, it becomes
% very easy to identify the peaks visually.
figure(3)
surf(theta, rho, H);
xlabel('theta','FontSize',16);
ylabel('rho','FontSize',16)
zlabel('Hough Transform counts','FontSize',16)

% From theta and rho and plot lines
lines = houghlines(fEdge,theta,rho,peaks,'FillGap',5,'MinLength',7); % remap the peaks into lines on top of the image.
figure(4), imshow(f), 
figure(4); hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
end
% See how 5 very identifyable DIRECTIONS in the image have been detected! -
% these define more than 5 segments, however, since several segments share the
% same line equation. 

% We can increase this number by increasing the numPeaks parameter in the
% houghpeaks function. (i.e., hughpeaks(H, 30))

%}

%{

% We can use a different transform to remap into the
% parameters of a circle equation. This is done with MATLAB's imfindcircles
% function

clear all; close all;
f = imread('assets/circles.tif');
[centers, radii] = imfindcircles(f,[6,100]); % We need to provide a fixed radius for the circle 
% equation, so it can look for the appropriate a and b parameters.
% In this case, matlab allows for a range, and it iterates between those
% different fixed radius values.


% Display the image
imshow(f);
hold on;

% Overlay detected circles
viscircles(centers, radii, 'EdgeColor', 'r');
hold off;

%}

%% TASK 4: SEGMENTATION BY THRESHOLDING

%{

% Using global thresholding 
f = imread('assets/yeast-cells.tif');
level = graythresh(f);
level_refined = graythresh(f) + 0.04; % Prevents the segmented areas to cluster different individual cells together. 
bw = imbinarize(f, level);
bw_refined = imbinarize(f, level_refined);
% montage({f, bw, bw_refined})

% ----------
% Using local thresholding:
neighborhoodSize = 65;
sensitivity = 0.5;

% Perform local thresholding
T = adaptthresh(f, sensitivity, 'NeighborhoodSize', [neighborhoodSize neighborhoodSize], 'Statistic', 'mean');
bw_local = imbinarize(f, T);

% Display results
montage({f, bw, bw_local})

%}

%% TASK 5: SEGMENTATION BY K-MEAN CLUSTERING


clear all; close all;
f = imread('assets/baboon.png');    % read image
[M N S] = size(f);                  % find image size
F = reshape(f, [M*N S]);            % resize as 1D array of 3 colours
% Separate the three colour channels 
R = F(:,1); G = F(:,2); B = F(:,3);
C = double(F)/255;          % convert to double data type for plotting
figure(1)
% scatter3(R, G, B, 1, C);    % scatter plot each pixel as colour dot
xlabel('RED', 'FontSize', 14);
ylabel('GREEN', 'FontSize', 14);
zlabel('BLUE', 'FontSize', 14);

% Find the centroids by the k-mean method
% perform k-means clustering
k = 10; % Segment in k=10 levels. It's going to find us 10 centers. 
[L,centers]=imsegkmeans(f,k);
% plot the means on the scatter plot
hold
figure(3);
scatter3(centers(:,1),centers(:,2),centers(:,3),100,'black','fill');


% display the segmented image along with the original
J = label2rgb(L,im2double(centers)); % The Matlab function labe2rgb turns 
% each element in L into the segmented colour stored in centers.

k2 = 3; % See what happens as we lower k: the image is segmented into fewer 
% levels, and greater regions. The centroids found are of course as many as k,
% and they take different values, maximinizing distances through the means. 
[L2,centers2]=imsegkmeans(f,k2);
% plot the means on the scatter plot
hold
figure(4);
scatter3(centers2(:,1),centers2(:,2),centers2(:,3),100,'black','fill');

% display the segmented image along with the original
J2 = label2rgb(L2,im2double(centers2));
figure(2)
% montage({f,J, J2})



%% TASK 6: SEGMENTATION USING THE WATERSHED METHOD



% Remember: the watershed method is very good to divide close by regions:
% it establishes "a dam" (i.e., a division line) whenever a 'water' would
% overflow to a neighboring region.

% Watershed segmentation with Distance Transform
clear all; close all;
I = imread('assets/dowels.tif');
f = im2bw(I, graythresh(I));
g = bwmorph(f, "close", 1);
g = bwmorph(g, "open", 1);
% we close and then open to remove small ireelevant details and
% imperfections.
montage({I,g});
title('Original & binarized cleaned image')

% calculate the distance transform image
gc = imcomplement(g); % we need to invert it sind bwdist will work with 
% what the software understands as the foreground pixels. 
D = bwdist(gc); % bwdist() computes distances from foreground pixels (1s) 
% to the nearest background (0s).
figure(2)
imshow(D,[min(D(:)) max(D(:))])
title('Distance Transform')

% The further you are from an object (background), the brighter the pixel value (greater distance).
% Peaks (brightest areas) correspond to the centers of the gaps between objects.

% Apply watershed to the distance transform image
L = watershed(imcomplement(D));
figure(10)
imshow(L, [0 max(L(:))])
title('Watershed Segemented Label')

% Merge everything to show segmentation
W = (L==0);
g2 = g | W;
figure(11)
montage({I, g, W, g2}, 'size', [2 2]);
title('Original Image - Binarized Image - Watershed regions - Merged dowels and segmented boundaries')

% Watershed segmentation is like flooding a topographic surface:
% 
% - We picture the distance transform as a 3D surface where peaks are high points (i.e., the center of the gaps between objects) 
% and edges around objects are valleys.
% - We place water sources (markers) at the highest points (local maxima).
% - Water floods the surface, and boundaries are formed where different water sources meet.
% - These boundaries correspond to the edges between objects.