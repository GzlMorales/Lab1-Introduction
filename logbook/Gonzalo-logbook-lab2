# Lab 2 | 23/01/2025 | Notes

## Task 1: Comments
Our blind spot exists because in that region of our eye, instead of photoreceptors, we have the optic nerve coming out of the eyeball. What’s interesting is that we do not notice such a blind spot because our brain interpolates or guesses what should be in that blind region by looking at what is around it. This way, it can trick us into thinking we are actually receiving visual information from that region when it is actually made up and based on inference.

## Task 2: Comments
For people with “color blindness,” their cones biased towards the long and medium wavelengths of the spectrum are too similar, so their reads make it confusing.

## Task 3: Comments
This aligns with the opponent process theory — the green-yellow channel gets tired out of being constantly excited by the signals coming from the green and yellow light, so our brain starts to normalize the values it receives. When we switch our sight to a white surface, the cones stop receiving those signals, and that is now interpreted as an active light signal in the opposite direction of the spectrum (the inverse of yellow in the spectrum is blue, and the inverse of green is red). This phenomenon is called color adaptation.

## Task 4: Comments
This is another example of color adaptation explained by the opponent process theory. After 20 seconds, due to neural adaptation, our brain starts ignoring the pink signal and instead only reacts to when such pink light is gone, thus interpreting that shift as green light (by subtracting that wavelength from the perceived color).

## Task 5: Comments
**a)** Our brain processes vision in 3 dimensions. Even when we see this illustration on a 2D screen showing these two rhomboid shapes of equal dimensions, we process the image as a depiction of a 3-dimensional scene. Thus, we interpret those oblique lines as longer than the horizontal ones because, in our 3D world, we perceive dimensions going towards the vanishing point as smaller.  

**b)** They have equal luminance, but we perceive A as darker because of how our brain processes light. The same way we process 2D images as 3-dimensional, we process flat, equally lit images as lit-up scenes, so our brain is capable of taking that lighting into account to interpret color.

## Task 6: Comments
**Lateral inhibition:** The intensity at a particular point is not rendered by a single photoreceptor but also by a group of receptors in its neighborhood.

## Task 7: Comments
This is produced by the irradiation illusion: high luminance regions are processed as bigger in size than darker ones. With an image with such high contrast between the tiles as this one, our brain is tricked into thinking the lines are not straight since it renders the size of the tiles as different.

## Task 8: Comments
Our brain interprets 2D depictions in three dimensions, even when it lacks cues for depth. This forces our brain to interpret 3D motion in a specific way, even when there is no particular sign of whether the dancer spins in one way or another.

## Task 9: Comments
This is an example of the Gestalt principle of continuation. When we perceive and interpret these visuals, our brain likes to look for incomplete patterns and complete missing details.

---

## Tasks 9-13 (MATLAB Code)

```matlab
imfinfo('peppers.png')
RGB = imread('peppers.png');  
imshow(RGB)

I = rgb2gray(RGB);
figure              % start a new figure window
imshow(I)

imshowpair(RGB, I, 'montage')
title('Original colour image (left) grayscale image (right)');

[R,G,B] = imsplit(RGB);
montage({R, G, B},'Size',[1 3])

HSV = rgb2hsv(RGB);
[H,S,V] = imsplit(HSV);
montage({H,S,V}, 'Size', [1 3])

XYZ = rgb2xyz(RGB);
[X,Y,Z] = imsplit(XYZ);
montage({X,Y,Z}, 'Size', [1 3])
