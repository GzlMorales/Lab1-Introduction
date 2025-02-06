%% 1. EROSION AND DILATION

%{

clear all
close all

A = imread('assets/text-broken.tif');
B1 = [0 1 0;
     1 1 1;
     0 1 0];    % create structuring element of type B II. 
A1 = imdilate(A, B1); % dilating returns a foreground value (i.e., 1), if ANY of the pixels within the B1 overlay are white. 
% montage({A,A1})

B2 = ones(3,3); % create structuring element of type B I.
A2 = imdilate(A, B2); % dilating returns a foreground value (i.e., 1), if ANY of the pixels within the B2 overlay are white. 
% montage({A,A2})

Bx = [1 0 1;
      0 1 0;
      1 0 1];
Ax = imdilate(A, Bx); % dilating returns a foreground value (i.e., 1), if ANY of the pixels within the Bx overlay are white. 
% montage({A,Ax})

% Applying the morphological operator twice will further thicken it:
A1_2 = imdilate(A1, B1); % Dilating the already dilated image,
% montage({A,A1_2})

%}


%% CREATING STRUCTURING ELEMENT WITH THE STREL FUNCTION, INSTEAD OF CONSTRUCTING A MATRIX MANUALLY.

%{

clear all
close all

SE = strel('disk',4);
SE.Neighborhood;         % print the SE neighborhood contents - we can see its a circle with a radious of 4 elements (1 1 1 1), and a diameter of 7 (1 1 1 1 1 1 1).

clear all
close all
A = imread('assets/wirebond-mask.tif');
SE2 = strel('disk',2);
SE10 = strel('disk',10);
SE20 = strel('disk',20);
SE20.Neighborhood; % print the E20 neighborhood contents. See how big it is! That's why it erodes so much.
E2 = imerode(A,SE2);
E10 = imerode(A,SE10);
E20 = imerode(A,SE20);
montage({A, E2, E10, E20}, "size", [2 2]) % We can see the foreground (white) gets smaller as the size of the structuring element with which we erode gets bigger.

%}

%% 2. OPENING OPERATION: i.e., SUBSEQUENT EROSION AND DILATION. 

%{

f = imread('assets/fingerprint-noisy.tif');
SE = ones(3,3); % create structuring element of type B I and size 3x3.
fe = imerode(f,SE); % erode image with SE
fed = imdilate(fe,SE); % dilate the already eroded image, i.e., open f. 

fo = imopen(f, SE); % open f using the built-in function.

% montage({f, fe, fed, fo}, "size", [2 2]) % Note how fed and fo are the same, since it the same operation!

foc = imclose(fo, SE);
% montage({f, fo, foc}, "size", [2 2]) % Note how the close operation "closes" those small gaps in between, since it starts with a dilation operation and then erodes.

% compare it with a gaussian operation
w_gauss = fspecial('Gaussian', [20 20], 0.5); % Gaussian filter, with sigma = 0.5. Note how the coefficients are adding to 1.0 so we preserve intensity.
g_gauss = imfilter(f, w_gauss, 0);
montage({f, fo, foc, g_gauss}) % the gaussian spatial filter also manages to remove the noise if we make the kernell big enough in size, but it also loses the sharpness of the fingerprint's geometry. 

%}

%% 3. EDGE DECTION AND WORKING WITH GRAYSCALE IMAGES

%{

% Thresholding. 
clear all
close all
I = imread('assets/blobs.tif');
I = imcomplement(I); % invert image, so the blobs are white (i.e. foreground).
level = graythresh(I); % finds the threshold that will binarize the image best, i.e., the value that would divide it into two groups of pixels ('light ones' and 'dark ones') within which the variance of luminosity would be minimized.
BW = imbinarize(I, level); % binarize image, i.e., it becomes 0s and 1s. 

% montage({I, BW}) % see result

% Perform boundary operation. i.e., we first erode it, and then subtract
% that from the original.

SE = ones(3,3); % create structuring element of type B I and size 3x3.
BW_eroded = imerode(BW,SE); % erode image with SE
BW_boundary = BW - BW_eroded; 
montage({I, BW, BW_eroded, BW_boundary}); % See how it detects the edges of the holes! Those edges are shown with the thickness of the amount eroded by the previous operation, which in turn is determined by the size of the structuring element. Note how the erode operation made the holes smaller. This is why, when we subtract this from the original BW, it shows the edges.

%}

%% 4. USING MATLAB'S BWMORPH FUNCTION TO CALL DIFFERENT MORPHOLOGICAL OPERATIONS ON B/W IMAGES

%{

% Syntax for this function: g = bwmorph(f, operation, n), where f is the
% image, operations is the name of the operation (string type argument),
% and n is the number of times the operation should be applied (default: 1)


f = imread('assets/fingerprint.tif');
f_i = imcomplement(f); % invert the image so the features of the fingerprint become the background (i.e., light and not dark).
level = graythresh(f_i); % finds the threshold that will binarize the image best, i.e., the value that would divide it into two groups of pixels ('light ones' and 'dark ones') within which the variance of luminosity would be minimized.
f_bw = imbinarize(f_i, level); % binarize image, i.e., it becomes 0s and 1s. 


% montage({f, f_bw}) % see result

% Perform thinning operations using the bmorph function

g1 = bwmorph(f_bw, 'thin');

% montage({f_bw, g1}) % see result. Notice how now the lines in the fingerprint are thinner and more easily distinguishable.

g2 = bwmorph(f_bw, 'thin', 2);
g3 = bwmorph(f_bw, 'thin', 3);
g4 = bwmorph(f_bw, 'thin', 4);

%montage({g1, g2, g3, g4}, 'size', [2 2]) % see result. Notice how these lines get progressively thinner as we apply the operation iteratively. g4 is losing too much detail. There's an optimal level of processing and we need to gauge that for each image.

% if i keep thinning the image, it will reach a stable point where we can
% only see very thin lines: 
ginf = bwmorph(f_bw, 'thin', inf);
% montage({f_bw, ginf}) % see result

% We can see that thickenning is the morphological dual of thinning: if we
% use the image with a black fingerprint on a white background, and perform
% a thickening operation, the result should do the same the shape as thinning the
% image with a white fingerprint over a black background (only inverted)

level_2 = graythresh(f);
f_bw_2 = imbinarize(f, level);

g1_2 = bwmorph(f_bw_2, 'thicken');

% montage({g1, g1_2}) % see result. Same result regarding shape! Only
% inverted

ginf_2 = bwmorph(f_bw_2, "thicken", inf);

montage({ginf, ginf_2}) % see result. Same result regarding shape! Only 
% inverted. It is actually a bit easier to see the result over a white
% background.

%}

%% 5. CONNECTED COMPONENTS

%{

t = imread('assets/text.png');
%imshow(t)
CC = bwconncomp(t) % find connected components in image t. This function 
% returns an object with the following fields: 
% > Connectivity: type of connectivity used, 
% > ImageSize: original image size, 
% > NumObjects: number of connected components detected,
% > PixelIdxList: cell array with as many cells as objects detected, where
% the k-th element is a vector containing the linear indices of the k-th 
% conncected component detected. 

numPixels = cellfun(@numel, CC.PixelIdxList); % cellfun is a matlab function
% to apply another function to each cell of a cell array. Such function to
% be applied to each cell must be given as an argument (in this case, the function 'numel')
% , and it must be preceded with @ since the argument is not function itself
% but its pointer. 
% 
% Of course, we also need to provide the cell array as argument, to let the
% code know to which cells it needs to apply the function.
%
% The @numel function is a function returns the number of pixels in each connected component. 

[biggest, idx] = max(numPixels); % Detect the biggest element with its index position in the array numpixels. 
t(CC.PixelIdxList{idx}) = 0; % Remove it calling it from its index in the PixelIdxList (turn the luminosity values 0)
figure
imshow(t)


%}

%% 6. MORPHOLOGICAL RECONSTRUCTION

%{

% It's method than can allow for recovering the details that an opening/closing 
% operation has lost as we eroded and then dilated or viceversa. However, for that,
% we need to provide a marker image, to indicate where the shape we want to
% reconstruct is, a mask to indicate those shapes, and a structuring element. 

clear all
close all
f = imread('assets/text_bw.tif');
se = ones(17,1); % tall structuring element, so eroding will only preserve a hint of the letters that are tall, signaling their position, and will output black everywhere else.
g = imerode(f, se); 
fo = imopen(f, se); % perform open to compare - since 'open' starts with an
% eroding operation, opening will also remove all other characters that are 
% not tall, but then the size of those tall elements will be somewhat preserved 
% with the subsequent dilation operation using the same structuring element.
fr = imreconstruct(g, f); % Now we can reconstruct those characters that survived using the original image f as a mask. 
% montage({f, g, fo, fr}, "size", [2 2])

% What if we reconstruct based on the opened image, instead of the eroded?
fr_2 = imreconstruct(fo, f); % Now we can reconstruct those characters that survived using the original image f as a mask.
% montage({g, fo, fr, fr_2}, "size", [2 2]) % we can see that the result is the same, since the marker image indicates the same positions for those elements!!

% Testing MATLAB's imfill function to fill the holes in the image
ff = imfill(f);
figure
montage({f, ff}) % It fills the holes within the characters. 

%}

%% 7. WORKING WITH A GRAYSCALE IMAGE AND GRAYSCALE OPERATIONS

% The dilation and erosion operations work in a similar fashion to their binary counterparts, 
% but instead of outputing 1 based on whether all pixels are 1 (strict requirement, erosion) or
% any are 1 (loose requirement, dilation), it selects the minimum value
% (strict requirement, erosion) or the maximum value (loose requirement,
% dilation).

clear all; close all;
f = imread('assets/headCT.tif');
se = strel('square',3);
gd = imdilate(f, se); 
ge = imerode(f, se);
gg = gd - ge; % edge detection subtracting the dilated result minus the 
% eroded result. We could also get the edges (albeit thinner and less 
% noticeable, by subtracting the eroded image from the original image, or 
% by subtracting the original image from the dilated image).

montage({f, gd, ge, gg}, 'size', [2 2])