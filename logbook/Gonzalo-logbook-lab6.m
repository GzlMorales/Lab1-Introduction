%% TASK 1: IMAGE RESIZING. SUBSAMPLING METHODS AND AVOIDING ALIASING

%{
clear all
close all
f = imread('assets/cafe_van_gogh.jpg');


% Create an image pyramid by successively downsampling the image
level{1} = f;  % Original image at level 1 (we cannot call it level 0 since arrays in matlab start at 1)
for i = 2:6
    % Subsequently subdivide the previous level in 2 -> 1/2, (1/2 * 1/2) = 1/4, (1/4 * 1/2) = 1/8... 
    level{i} = level{i-1}(1:2:end, 1:2:end, :); % Drop every other row & column
end

% Display images as a 2x3 montage
figure;
montage({level{1}, level{2},level{3},level{4},level{5},level{6}}, 'Size',[2,3]); 

% Result: we can see, as the image is scaled down. more and more unwanted
% artifacts appear that were not in fact in the original image. This is
% called aliasing, and it eventually occurs whenever we try to subsequently 
% subsample/scale down high-detail images simply by dropping pixels instead
% of applying a low-pass filter first, like a Gaussian filter.

% Let's try resizing using gaussian and then dropping pixels, by using
% Matlab's inbuilt resize function:

level_new{1} = f;  % Original image at level 1 (we cannot call it level 0 since arrays in matlab start at 1)
for i = 2:6
    % Subsequently subdivide the previous level in 2 -> 1/2, (1/2 * 1/2) = 1/4, (1/4 * 1/2) = 1/8... 
    level_new{i} = imresize(level_new{i-1}, 0.5);
end

figure;
montage({level_new{1}, level_new{2},level_new{3},level_new{4},level_new{5},level_new{6}}, 'Size',[2,3]); 
% The result is much more similar to the original image, only smaller and
% hence losing detail as we scale down (i.e. as we go up the pyramid). We
% don't see any artifacts or highly detailed elements that were not
% originally present, since these have been avoided by first applying the
% gaussian filter through Matlab's inbuilt function.

%}

%% TASK 2: PATTERN MATCHING AND NORMALIZED CROSS-CORRELATION

%{

% Intro: cross correlation is the same as applying a an image filter using
% the template of what we're looking for (w) as the kernel. This, when overlaid
% on top of the occurence of said pattern in the image will return a very
% high value (as long as said occurence is in the same scale and rotation).
% 

% Normalized cross correlation is the result of normalizing the results of
% applying such filter so that they range from [-1,1], 1 being a perfect
% match, and -1 being a total mismatch.  

clear all; close all;
f = imread('assets/salvador_grayscale.tif');
w = imread('assets/template2.tif');
c = normxcorr2(w, f);
figure(1)
surf(c)
shading interp % We can see there is a very clear peak, which will 
% correspond to the occurence of our template in the image (in the same 
% scale and orientation)

% Find the exact location of said peak:

[ypeak, xpeak] = find(c==max(c(:))); % finds the coordinates of the maximum 
% value within the c map (c being the values of normalized cross 
% correlation for each pixel in the image, mapped in X and Y)

% Coordinate correction to find the top-left corner of the rectangle that
% will frame our match. 
yoffSet = ypeak-size(w,1);
xoffSet = xpeak-size(w,2);
figure(2)
imshow(f)
drawrectangle(gca,'Position', ... % This '...' is just to split the line of code
[xoffSet,yoffSet,size(w,2),size(w,1)], 'FaceAlpha',0);

%}

%% TASK 3: FEATURE DETECTION USING SIFT

%{

% Intro: SIFT stands for Scale-invariant feature transform -- it's an
% algorithm that allows us to transform any feature in the image to have
% the same orientation, scale, intensity, etc. than that of a template, so
% that it can be used to detect features in an image independently from
% these factors. In other words, it allows us to detect features in an
% image even when these are not exact matches (pixel-by-pixel) of a
% template, but instead may be rotated, have a different size, etc.

clear all; close all;
I = imread('assets/salvador.tif');
f = im2gray(I); % Sift works in black and white, with values of luminance. 
points = detectSIFTFeatures(f); % This uses an matlab's inbuilt feature 
% (within the computer vision toolbox add-on) to apply SIFT to detect 
% 'interesting' image features / blobs. These are found as it checks for
% the differnce of guassians within subregions of different sizes,
% selecting those that show the highest peaks. 

figure(1); imshow(I);
hold on;
plot(points.selectStrongest(100));

% Note that this just detects interesting features in the image, but it
% does not find a particular feature that we may be looking for. For this,
% we need to do SIFT feature matching (next exercise).

%}

%% TASK 4: FEATURE MATCHING USING SIFT

%{

clear all; close all;
I1 = imread('assets/cafe_van_gogh.jpg');
I2 = imresize(I1, 0.5); % Smaller image (@scale 1/2)
f1 = im2gray(I1);
f2 = im2gray(I2);
points1 = detectSIFTFeatures(f1);
points2 = detectSIFTFeatures(f2);
Nbest = 100;
bestFeatures1 = points1.selectStrongest(Nbest);
bestFeatures2 = points2.selectStrongest(Nbest);
figure(1); imshow(I1);
hold on;
plot(bestFeatures1);
hold off;
figure(2); imshow(I2);
hold on;
plot(bestFeatures2);
hold off;

% Applying SIFT to the smaller version of the image finds better
% 'interesting features' than doing so with the originally-sized image.
% This may be so because the original image has a high level of detail that
% makes the algortihm focus on smaller elements:

% > Less Noise: High-resolution images may contain noise that can introduce 
% false keypoints. Scaling down reduces noise and focuses on more prominent features.
% > Better for Coarse Features: If the image contains large, dominant features, 
% reducing the resolution may help SIFT detect them more effectively.

% Additionally, performance is better since the algorithm needs to work
% with less data.

% Let's see how these points match between these two results, and how
% we're able to detect them, independent of the scale. 

[features1, valid_points1] = extractFeatures(f1, points1);
[features2, valid_points2] = extractFeatures(f2, points2);

 indexPairs = matchFeatures(features1, features2, 'Unique', true);

 matchedPoints1 = valid_points1(indexPairs(:,1),:);
 matchedPoints2 = valid_points2(indexPairs(:,2),:);

 figure(3);
 showMatchedFeatures(f1,f2,matchedPoints1,matchedPoints2);

% Now, we can compare how some of these 'best points' that we found (which
% are not exactly the same set, as we discussed before) are matched,
% independently of the scale.

[best_features1, best_valid_points1] = extractFeatures(f1, bestFeatures1);
[best_features2, best_valid_points2] = extractFeatures(f2, bestFeatures2);

 indexPairs = matchFeatures(best_features1, best_features2, 'Unique', true);

 best_matchedPoints1 = best_valid_points1(indexPairs(:,1),:);
 best_matchedPoints2 = best_valid_points2(indexPairs(:,2),:);

 figure(3);
 showMatchedFeatures(f1,f2,best_matchedPoints1,best_matchedPoints2);

 % We can see how some do match!

%}

 %% TASK 5: OBJECT TRACKING USING SIFT AND TURF

clear all; close all;
I1 = imread('assets/traffic_1.jpg');
I2 = imread('assets/traffic_2.jpg');
f1 = im2gray(I1);
f2 = im2gray(I2);
points1 = detectSIFTFeatures(f1);
points2 = detectSIFTFeatures(f2);
Nbest = 300;
bestFeatures1 = points1.selectStrongest(Nbest);
bestFeatures2 = points2.selectStrongest(Nbest);


[features1, valid_points1] = extractFeatures(f1, bestFeatures1);
[features2, valid_points2] = extractFeatures(f2, bestFeatures2);

indexPairs = matchFeatures(features1, features2, 'Unique', true);

matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);

figure(3);
showMatchedFeatures(f1,f2,matchedPoints1,matchedPoints2);

% We can see it manages to find matches not only within objects that have
% not moved in the scene (i.e., certain road features and static elements
% like the division wall), but also for moving objects (like the vehicles)
% as long as the image has not changed too much (some cars have changed a 
% lot due to perspective and those are not found).

% Let's try the same algorithm using SURF instead of SIFT:

I1 = imread('assets/traffic_1.jpg');
I2 = imread('assets/traffic_2.jpg');
f1 = im2gray(I1);
f2 = im2gray(I2);
points1 = detectSURFFeatures(f1);
points2 = detectSURFFeatures(f2);
Nbest = 300;
bestFeatures1 = points1.selectStrongest(Nbest);
bestFeatures2 = points2.selectStrongest(Nbest);


[features1, valid_points1] = extractFeatures(f1, bestFeatures1);
[features2, valid_points2] = extractFeatures(f2, bestFeatures2);

indexPairs = matchFeatures(features1, features2, 'Unique', true);

matchedPoints1 = valid_points1(indexPairs(:,1),:);
matchedPoints2 = valid_points2(indexPairs(:,2),:);

figure(3);
showMatchedFeatures(f1,f2,matchedPoints1,matchedPoints2);

% SURF is very good and advanced at tracking objects! It is more tolerant
% than SIFT to changes in perspective, being able to still track the silver
% car even as its view has substantially channged as it moved away. 

%% TASK 6: IMAGE RECOGNITION USING NEURAL NETWORKS

% Object recognition using webcam and various neural network models

camera = webcam;                            % create camera object for webcam
net = google;                               % choosing neural network
inputSize = net.Layers(1).InputSize(1:2);   % find neural network input size. 
                                            % Note: these neural networks
                                            % are multi-layered (hence the 
                                            % 'deep learning' term).
                                            % Because of this, we need to
                                            % find the input size of the
                                            % entry layer (i.e., Layer with
                                            % index 1).
figure 
I = snapshot(camera);      
image(I);
f = imresize(I, inputSize);                 % resize image to match network.
                                            % This resizes avoiding
                                            % aliasing by first applying a
                                            % gaussian filter.

                                            % Keeping track of time using
                                            % tic, toc.
tic;                                        % mark start time
[label, score] = classify(net,f);           % classify f with neural network net
toc                                         % report elapsed time

title({char(label), num2str(max(score),2)}); % label object