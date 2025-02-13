# Lab 4 | Morphological Image Processing | 06/02/2025 | Notes

> Please note: notes and learnings on the operations carried out can be found in the  comments within the code snippets. This is done this way to interrupt the code as little as possible while having a synchronized view of the learnings and reflections. 

## 1. Erosion and Dilation
### Dilation

```matlab
clear all
close all

A = imread('assets/text-broken.tif');
B1 = [0 1 0;
     1 1 1;
     0 1 0];    % create structuring element of type B II. 
A1 = imdilate(A, B1); % dilating returns a foreground value (i.e., 1), if ANY of the pixels within the B1 overlay are white. 
montage({A,A1})
```
![s18383302132025](https://a.okmd.dev/md/67ae3c2b62399.png)

```matlab
B2 = ones(3,3); % create structuring element of type B I.
A2 = imdilate(A, B2); % dilating returns a foreground value (i.e., 1), if ANY of the pixels within the B2 overlay are white. 
montage({A,A2})
```
![s18384402132025](https://a.okmd.dev/md/67ae3c34e306a.png)

```matlab
Bx = [1 0 1;
      0 1 0;
      1 0 1];
Ax = imdilate(A, Bx); % dilating returns a foreground value (i.e., 1), if ANY of the pixels within the Bx overlay are white. 
montage({A,Ax})
```
![s18385602132025](https://a.okmd.dev/md/67ae3c4244ace.png)

```matlab
% Applying the morphological operator twice will further thicken it:
A1_2 = imdilate(A1, B1); % Dilating the already dilated image,
montage({A,A1_2})
```
![s18391702132025](https://a.okmd.dev/md/67ae3c562f689.png)


### Creating Structuring Element with the Strel Function

```matlab
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
```
![s18394002132025](https://a.okmd.dev/md/67ae3c6d816d5.png)

## 2. Opening Operation: Erosion + Dilation

```matlab
f = imread('assets/fingerprint-noisy.tif');
SE = ones(3,3); % create structuring element of type B I and size 3x3.
fe = imerode(f,SE); % erode image with SE
fed = imdilate(fe,SE); % dilate the already eroded image, i.e., open f. 

fo = imopen(f, SE); % open f using the built-in function.

% montage({f, fe, fed, fo}, "size", [2 2]) % Note how fed and fo are the same, since it the same operation!
```
![s18402002132025](https://a.okmd.dev/md/67ae3c9661846.png)

```matlab

foc = imclose(fo, SE);
montage({f, fo, foc}, "size", [2 2]) % Note how the close operation "closes" those small gaps in between, since it starts with a dilation operation and then erodes.
```
![s18403002132025](https://a.okmd.dev/md/67ae3ca00fcb5.png)

```matlab
% compare it with a gaussian operation
w_gauss = fspecial('Gaussian', [20 20], 0.5); % Gaussian filter, with sigma = 0.5. Note how the coefficients are adding to 1.0 so we preserve intensity.
g_gauss = imfilter(f, w_gauss, 0);
montage({f, fo, foc, g_gauss}) % the gaussian spatial filter also manages to remove the noise if we make the kernell big enough in size, but it also loses the sharpness of the fingerprint's geometry.
```
![s18403802132025](https://a.okmd.dev/md/67ae3ca852753.png)

## 3. Edge Detection and Grayscale Images

```matlab
% Thresholding. 
clear all
close all
I = imread('assets/blobs.tif');
I = imcomplement(I); % invert image, so the blobs are white (i.e. foreground).
level = graythresh(I); % finds the threshold that will binarize the image best, i.e., the value that would divide it into two groups of pixels ('light ones' and 'dark ones') within which the variance of luminosity would be minimized.
BW = imbinarize(I, level); % binarize image, i.e., it becomes 0s and 1s. 

montage({I, BW}) % see result
```
![s18405002132025](https://a.okmd.dev/md/67ae3cb364bdf.png)


```matlab
% Perform boundary operation. i.e., we first erode it, and then subtract
% that from the original.

SE = ones(3,3); % create structuring element of type B I and size 3x3.
BW_eroded = imerode(BW,SE); % erode image with SE
BW_boundary = BW - BW_eroded; 
montage({I, BW, BW_eroded, BW_boundary}); % See how it detects the edges of the holes! Those edges are shown with the thickness of the amount eroded by the previous operation, which in turn is determined by the size of the structuring element. Note how the erode operation made the holes smaller. This is why, when we subtract this from the original BW, it shows the edges.

```
![s18410002132025](https://a.okmd.dev/md/67ae3cbe6b89f.png)

## 4. Thinning with bwmorph

```matlab
% Syntax for this function: g = bwmorph(f, operation, n), where f is the
% image, operations is the name of the operation (string type argument),
% and n is the number of times the operation should be applied (default: 1)


f = imread('assets/fingerprint.tif');
f_i = imcomplement(f); % invert the image so the features of the fingerprint become the background (i.e., light and not dark).
level = graythresh(f_i); % finds the threshold that will binarize the image best, i.e., the value that would divide it into two groups of pixels ('light ones' and 'dark ones') within which the variance of luminosity would be minimized.
f_bw = imbinarize(f_i, level); % binarize image, i.e., it becomes 0s and 1s. 


montage({f, f_bw}) % see result
```
![s18411002132025](https://a.okmd.dev/md/67ae3cc807aeb.png)

```matlab
% Perform thinning operations using the bmorph function

g1 = bwmorph(f_bw, 'thin');

montage({f_bw, g1}) % see result. Notice how now the lines in the fingerprint are thinner and more easily distinguishable.
```
![s18414502132025](https://a.okmd.dev/md/67ae3ceab7ae2.png)

```matlab
g2 = bwmorph(f_bw, 'thin', 2);
g3 = bwmorph(f_bw, 'thin', 3);
g4 = bwmorph(f_bw, 'thin', 4);

montage({g1, g2, g3, g4}, 'size', [2 2]) % see result. Notice how these lines get progressively thinner as we apply the operation iteratively. g4 is losing too much detail. There's an optimal level of processing and we need to gauge that for each image.
```
![s18415302132025](https://a.okmd.dev/md/67ae3cf2ad8ea.png)

```matlab
% if i keep thinning the image, it will reach a stable point where we can
% only see very thin lines: 
ginf = bwmorph(f_bw, 'thin', inf);
montage({f_bw, ginf}) % see result. We can see that thickenning is the morphological dual of thinning: if % we use the image with a black fingerprint on a white background, and perform a thickening operation, the % result should do the same the shape as thinning the image with a white fingerprint over a black % background (only inverted).
```
![s18420402132025](https://a.okmd.dev/md/67ae3cfd042b3.png)

```matlab
level_2 = graythresh(f);
f_bw_2 = imbinarize(f, level);

g1_2 = bwmorph(f_bw_2, 'thicken');

montage({g1, g1_2}) % see result. Same result regarding shape! Only
% inverted
```
![s18421802132025](https://a.okmd.dev/md/67ae3d0aca365.png)

```matlab
ginf_2 = bwmorph(f_bw_2, "thicken", inf);

montage({ginf, ginf_2}) % see result. Same result regarding shape! Only 
% inverted. It is actually a bit easier to see the result over a white
% background.
```
![s18422702132025](https://a.okmd.dev/md/67ae3d14073e4.png)

## 5. Connected components

```matlab
t = imread('assets/text.png');
%imshow(t)
CC = bwconncomp(t) % find connected components in image t. 
```
This function returns an object with the following fields: 
- Connectivity: type of connectivity used, 
- ImageSize: original image size, 
- NumObjects: number of connected components detected,
- PixelIdxList: cell array with as many cells as objects detected, where
 the k-th element is a vector containing the linear indices of the k-th 
 conncected component detected. 

```matlab
numPixels = cellfun(@numel, CC.PixelIdxList); % Get size (number of pixels) for each connected compenents
```
 cellfun is a matlab function to apply another function to each cell of a cell array. Such function to be applied to each cell must be given as an argument (in this case, the function 'numel'), and it must be preceded with @ since the argument is not function itself but its pointer. 

Of course, we also need to provide the cell array as argument, to let the code know to which cells it needs to apply the function.The @numel function is a function returns the number of pixels in each connected component. 

```matlab
[biggest, idx] = max(numPixels); % Detect the biggest element with its index position in the array numpixels.

t(CC.PixelIdxList{idx}) = 0; % Remove it calling it from its index in the PixelIdxList (turn the luminosity values 0)

figure
imshow(t)
```

![s18424202132025](https://a.okmd.dev/md/67ae3d2353efb.png)

## 6. Morphological reconstruction

It's a method than can allow for recovering the details that an opening/closing operation has lost as we eroded and then dilated or viceversa. However, for that, we need to provide a marker image, to indicate where the shape we want to reconstruct is, a mask to indicate those shapes, and a structuring element.

```matlab

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
montage({f, g, fo, fr}, "size", [2 2])
```
![s18425602132025](https://a.okmd.dev/md/67ae3d31c5f4f.png)

```matlab
% What if we reconstruct based on the opened image, instead of the eroded?
fr_2 = imreconstruct(fo, f); % Now we can reconstruct those characters that survived using the original image f as a mask.
% montage({g, fo, fr, fr_2}, "size", [2 2]) % we can see that the result is the same, since the marker image indicates the same positions for those elements!!
```
![s18430902132025](https://a.okmd.dev/md/67ae3d3eb1496.png)

```matlab
% Testing MATLAB's imfill function to fill the holes in the image
ff = imfill(f);
figure
montage({f, ff}) % It fills the holes within the characters. 
```

![s18433402132025](https://a.okmd.dev/md/67ae3d5873328.png)

## 7. Working with a grayscale image and grayscale operations

The dilation and erosion operations work in a similar fashion to their binary counterparts, but instead of outputing 1 based on whether all pixels are 1 (strict requirement, erosion) or any are 1 (loose requirement, dilation), it selects the minimum value (strict requirement, erosion) or the maximum value (loose requirement, dilation).

```matlab

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
```

![s18440602132025](https://a.okmd.dev/md/67ae3d7785cba.png)

## Challenges

### 1. Counting Fillings

Option #1: Removing the noise through gaussian filtering

```matlab
% 1. Open image
clear all; close all;
f = imread('assets/fillings.tif');
% imshow(f); % See output - very noisy image, hard to see cavities.

% 2. Reducing noise using a gaussian filter 
gauss_filter = fspecial('Gaussian', [50 50], 5); 
f_gauss = imfilter(f, gauss_filter, 0);
% montage({f, f_gauss}) 

% 3. Binarizing the image through thresholding. 
level = 0.85; % Manual fine tuning - fing the clearest output through try and error.
BW = imbinarize(f_gauss, level); % binarize image, i.e., it becomes 0s and 1s. 
montage({f, f_gauss, BW}) % see output. 

% 4. Counting the cavities (i.e., connected components in the binarized image)
CC = bwconncomp(BW);
num_cavities = CC.NumObjects

cavity_size = cellfun(@numel, CC.PixelIdxList) 
```

![s18443002132025](https://a.okmd.dev/md/67ae3d8fe2e23.png)

Option #2: Using grayscale dilation instead of gaussian filtering to remove noise. 


```matlab
% 1. Open image
clear all; close all;
f = imread('assets/fillings.tif');
% imshow(f); % See output - very noisy image, hard to see cavities.

% 2. Reducing noise using a dilation
se = strel('disk',2);
fd = imdilate(f,se);
% montage({f, fd}) 

% 3. Binarizing the image through thresholding. 
level = 0.93; % Manual fine tuning - fing the clearest output through try and error.
BW = imbinarize(fd, level); % binarize image, i.e., it becomes 0s and 1s. 
montage({f, fd, BW}) % see output.

% 4. Counting the cavities (i.e., connected components in the binarized image)
CC = bwconncomp(BW);
num_cavities = CC.NumObjects

cavity_size = cellfun(@numel, CC.PixelIdxList) 
```

![s18444002132025](https://a.okmd.dev/md/67ae3d99d0b42.png)

### 2. Extracting Main Palm Features

```matlab
% 1. Open image
clear all; close all;
f = imread('assets/palm.tif');
% imshow(f); % See output - image with lots of detail. Hard to see main
% palm features.

% 2. Grayscale erode operation
se = strel('square',5);
ge = imerode(f, se);


% Binarize through thresholding
i = imcomplement(ge); % Invert image so ther lines are clearer.
level = graythresh(i); % finds the threshold that will binarize the image best, i.e., the value that would divide it into two groups of pixels ('light ones' and 'dark ones') within which the variance of luminosity would be minimized.
bw = imbinarize(i, level); % binarize image, i.e., it becomes 0s and 1s. 
% montage({f, bw}) % see output. 

% Perform a close operation so it removes small, irrelevant details.
size = 3; % We fine tune the size of the structuring element through try and error to see which result gives us the right amount of details. 
se = ones(size,size); % structuring element for the morphological operation. 
fc = imclose(bw, se);
montage({f, ge, bw, fc}, "size", [2 2]) % Note how the close operation "closes" those small gaps in between.
```

![s18445102132025](https://a.okmd.dev/md/67ae3da5222fd.png)

### 3. Counting Red Blood Cells
The following two methods count blood cells that are entirely in the frame, either by:
- Using the imfill method to fill holes and count them by subtracting the result from the original image.
- Using the imclearborder method to remove the elements touching the edge of the frame.

Using imfill:
```matlab
clear all; close all;
f = imread('assets/normal-blood.png');
g = im2gray(f);
i = imcomplement(g);
% imshow(f); % See output 

level = graythresh(i); % finds the threshold that will binarize the image best, i.e., the value that would divide it into two groups of pixels ('light ones' and 'dark ones') within which the variance of luminosity would be minimized.
bw = imbinarize(i, level); % binarize image, i.e., it becomes 0s and 1s.
% montage({f, bw}) % see output. 

% Close image to ensure holes are visible
size = 7;
se = ones(size,size); % structuring element for the morphological operation. 
bwc = imclose(bw, se);

no_holes = imfill(bwc, 26, 'holes'); % holes must be entirely in the frame for this function to fill them.

holes = no_holes - bwc;

montage({f, bwc, no_holes, holes}) % see output. 


CC = bwconncomp(holes);
num_holes = CC.NumObjects
```

![s18450102132025](https://a.okmd.dev/md/67ae3daf8c16f.png)

Using imclearborder
```matlab
clear all; close all;
f = imread('assets/normal-blood.png');
g = im2gray(f);
i = imcomplement(g);
% imshow(f); % See output ![![alt text](image-1.png)](image.png)

level = graythresh(i); % finds the threshold that will binarize the image best, i.e., the value that would divide it into two groups of pixels ('light ones' and 'dark ones') within which the variance of luminosity would be minimized.
bw = imbinarize(i, level); % binarize image, i.e., it becomes 0s and 1s.
% montage({f, bw}) % see output. 

% Close image to ensure holes are visible
size = 7;
se = ones(size,size); % structuring element for the morphological operation. 
bwc = imclose(bw, se);


clear_border = imclearborder(imcomplement(bwc));

CC = bwconncomp(clear_border);
num_holes = CC.NumObjects
% edge_items = bwc - clear_border;

montage({f, bwc, clear_border}) % see output.
```
![s18450902132025](https://a.okmd.dev/md/67ae3db7dac05.png)