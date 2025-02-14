# Lab 3 | Intensity transforms & Spatial filters | 30/01/2025 | Notes

## Task 1: Contrast enhancement with function imadjust

```matlab
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

g2 = imadjust(f, [0.5 0.75], [0 1]); % Remaps values within the 0.5-0.75 range to the extremes (0,1).
g3 = imadjust(f, [ ], [ ], 2); % Applies gamma correction with value 2. This achieves similar results but preserves more details because g2 simply truncated the histogram.
figure
montage({g2,g3})
```

## Task 2: Contrast-stretching transformation

```matlab
clear all       % clear all variables
close all       % close all figure windows
f = imread('assets/bonescan-front.tif');
r = double(f);  % uint8 to double conversion
k = mean2(r);   % find mean intensity of image - mean2 function computes mean in a 2D matrix.
E = 0.9;        % steepness of the contrast stretching function
s = 1 ./ (1.0 + (k ./ (r + eps)) .^ E); % r could be zero! That's why we add eps. eps is a Matlab variable for a very small value that avoids division by 0.
g = uint8(255*s);
imshowpair(f, g, "montage")
```

## Task 3: Contrast Enhancement using Histogram

```matlab
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
```

## Task 4: Noise reduction with lowpass filter

```matlab
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
```

## Task 5: Median filtering

```matlab
g_median = medfilt2(f, [7 7], 'zero');
figure; montage({f, g_median})

% Thoughts: the median filter is an interesting filter. It has
% no coefficients. We overlay it pixel by pixel, just like before.
% And then we observe the pixels in that mask overlay, and arrange
% them by value. Then we take the middle one. In other words: we find the pixel
% with the median value in that neighborhood, and we output that value!
```

## Task 6: Sharpening the image with Laplacian, Sobel and Unsharp filters

```matlab
clear all
close all
f = imread('assets/moon.tif');
imshow(f)

w_laplacian = fspecial("laplacian"); % Laplacian filter 
w_sobel = fspecial("sobel") % Sobel filter

g_laplacian = imfilter(f, w_laplacian, 0);
g_sharpened_laplacian = f + 5 * g_laplacian; % Added a 5x weight to sharpen even more.
g_sobel = imfilter(f, w_sobel, 0);
g_sharpened_sobel = f + g_sobel; 
g_unsharp = imsharpen(f); % Unsharp filter - please note: the name is confusing, but this filter actually sharpens the image.

figure
montage({f, g_sharpened_laplacian, g_sharpened_sobel, g_unsharp}, 'Size',[2,2])
```

![s11133102142025](https://a.okmd.dev/md/67af255e37f33.png)

**Laplacian filter:** Uses a kernel that extracts change: it outputs a big value when it detects a variation in intensity. In other words, it detects edges, and it intensifies them. This is a kind of high-pass filter. Original + Laplacian can give us a sharpenned image. And it may be useful to use a multiplying factor to increase that sharpness even further (in this case, we used a 5x coefficient)

**Sobel filter:** It uses a gradient image operation. Gx detects edges in the vertical direction. Gy detects edges in the horizontal direction. Now we can get the resulting image by either squaring them and square-rooting them, or using the absolute value. Like with the Laplacian filter, we need to add the output to the original image to get the sharpened image. The result is often more natural-feeling than that obtained with the Laplacian filter. 


*Please note:* an **unsharp filter** is in fact a type of sharpening filter. The term is used to differenciate between unsharp marking and highboost filtering -- they both operate the same way: 

- Blur the original image.
- Subtract the blurred output from the original, which results in a "mask".
- Add such mask to the original, with a certain weight coefficient.

If the weight is 1, it's an unsharp filter. Otherwise, if greater than 1, it's a highboost filter. 

## Task 7: Test yourself challenges

**Improving image contrast using histogram equalization:** 

```matlab
clear all
close all
f = imread('assets/lake&tree.png');

g = histeq(f,256); % histogram equalize f: Histogram equalization looks to make the intensity values in the histogram be as equally distributed as possible, enhancing contrast.
close all
montage({f, g})
figure;
```

Histogram equalization looks to make the intensity values in the histogram be as equally distributed as possible, "spreading out the histogram", and thus enhancing contrast. This transformation, of course, is dependent on the contents of the image itself, so it first needs to analyze the distribution of values in the original histogram and, based on a probability distribution function, distributes the values so they reach the extremes of the new histogram. 

![s11255702142025](https://a.okmd.dev/md/67af28481e36e.png)

**Edge detection using sobel and laplacian filters:**

```matlab
clear all
close all
f = imread('assets/circles.tif');

w_gauss = fspecial("gaussian", [40,40], 1.5); % Using a gaussian filter (i.e., low-pass), to blur out irrelevant details, like those of the wooden surface.
w_sobel = fspecial("sobel") % Sobel filter, to sharpen image at the edges.
w_laplacian = fspecial("laplacian"); % Laplacian filter with default alpha: 0.2. Alpha determines the 'shape' of the kernel (i.e., the contribution of the diagonal neighbors in the discrete approximation of the second derivative).


g_gauss = imfilter(f, w_gauss);
g_laplacian = imfilter(g_gauss, w_laplacian, 0);
g_sobel = imfilter(g_gauss, w_sobel, 0);
g_combined = g_sobel + 10 * g_laplacian; % We enhance edge detection by combining it with a laplacian filter.
g_unsharp = imsharpen(g_combined);


figure
montage({f, g_combined})
```

![s12045502142025](https://a.okmd.dev/md/67af3169cea4e.png)

**Exposure and contrast correction:**

```matlab
clear all
close all
f = imread('assets/office.jpg');
% imshow(f); % See result -> we can see it's a bit low contrast. Also, if we're trying to see the office and not the exterior, it's a bit underexposed.

[fmin, fmax] = bounds(f(:)) % get minimum and maximum values of intensity within the whole image
% we get 0 to 255 -- full range [0,255]. However, the distribution is
% probably not great: we can check seeing the PDF:

g_pdf = imhist(f) ./ numel(f);  % compute PDF - probability distribution function for the intensity function of the already remapped function g.
plot(g_pdf) % We can see it's a bit exposed to the left (i.e., a bit dark: we may get noise when we try to brighten up).
```
![s12415302142025](https://a.okmd.dev/md/67af3a12df7f3.png)

We can see it's a bit exposed to the left (i.e., a bit dark: we may get noise when we try to brighten up).

To fix this, we can try applying histogram equalization to enhance contrast and exposure.





```matlab
% Applying histogram equalization to enhance contrast and exposure
equalized = histeq(f,256); % histogram equalize f: Histogram equalization looks to make the intensity values in the histogram be as equally distributed as possible, enhancing contrast.
imshow(equalized);
```

![s12484602142025](https://a.okmd.dev/md/67af3bb151735.png)!

The result looks rather unnatural (too saturated and high-contrast). Alternatively, we could apply a simple gamma correction. 

```matlab
gamma_corrected = imadjust(f, [ ], [ ], 0.8); % Gamma correction with gamma < 1 = increased exposure.
imshow(gamma_corrected);
```
![s12431502142025](https://a.okmd.dev/md/67af3a65e2e28.png)

The result is better exposed, but too low contrast. We might want to try with a custom remapping of the histogram:

```matlab
% Manually adjust the histogram to adjust exposure to our liking.
manual_hist_remap = imadjust(f, [0.1 0.65], [0 1]);

````

The result is much better. But we can make it even better if we reduce the noise using a median filter.


```matlab
% Reduce noise using a median filter for each channel
% Split channels
R = manual_hist_remap(:,:,1);
G = manual_hist_remap(:,:,2);
B = manual_hist_remap(:,:,3);
% Apply median filter to each channel
R_filt = medfilt2(R, [3 3], 'zero');
G_filt = medfilt2(G, [3 3], 'zero');
B_filt = medfilt2(B, [3 3], 'zero');
% Reconstruct the image to apply RGB median filter
noise_reduced = cat(3, R_filt, G_filt, B_filt);

montage({f, equalized, manual_hist_remap, noise_reduced});
```

![s12431502142025](../img/s12475702142025.png)

