# Image Transformations: Rotation and Shearing

## Task 1: Image rotation

```matlab
testMatrix = [0,0.5,1; 0,0.5,1; 0,0.5,1]; % Used for testing and debugging
function [out] = rotateImage(in, theta) % For convenience, rotation angle "theta" is in degrees

    matrixSize = size(in);
    out = zeros(matrixSize); % The final dimensions of the canvas remain the same. We will simply crop out the corners of the image that are out of the original bounds when rotated.

    centerPixelCoordinate = [round(size(in, 1)/2); round(size(in, 2)/2)];

    rotationMatrix = [cosd(theta), sind(theta); -sind(theta), cosd(theta)]; % "theta" in degrees
    reverseMatrix = inv(rotationMatrix); % Reverse mapping avoids the holes in the transformed image

    for x = 1:size(in, 1)
        for y = 1:size(in, 2)

            % pixelValue = in(x, y); % No longer needed. I used to do:
            % out(newX,newY) = pixelValue but that led to holes in the
            % image for not using the reverse mapping properly.

            currentPixelCoordinate = [x;y];

            newPixelCoordinate = (reverseMatrix * (currentPixelCoordinate - centerPixelCoordinate)) + centerPixelCoordinate;
            newX = round(newPixelCoordinate(1));
            newY = round(newPixelCoordinate(2));

            % I already initialize all values to zeros, so I just need to
            % overwrite if the transformed pixel coordinates are within the
            % new image canvas.
            if (newX > 1 && newX <= size(in,1) && newY > 1 && newY <= size(in,2))
                out(x,y) = in(newX,newY);
            end

        end
    end

end

result = rotateImage(clown, 45);
imshow(result);
```

## Task 2: Image shearing

```matlab
function [out] = shearImage(in, xShear, yShear) % Values of xShear and yShear from 0 to 1, relative to the fraction of width and height of the canvas.

    matrixSize = size(in);
    out = zeros(matrixSize); % The final dimensions of the canvas remain the same. We will simply crop out the corners of the image that are out of the original bounds when sheared.

    centerPixelCoordinate = [round(size(in, 1)/2); round(size(in, 2)/2)];
    shearMatrix = [1, xShear; yShear, 1]; % Similar process to rotation, only now the transform matrix is a simple shear matrix.

    for x = 1:size(in, 1)
        for y = 1:size(in, 2)

            currentPixelCoordinate = [x;y];
            newPixelCoordinate = (shearMatrix * (currentPixelCoordinate - centerPixelCoordinate)) + centerPixelCoordinate;

            newX = round(newPixelCoordinate(1));
            newY = round(newPixelCoordinate(2));
            if newX > 0 && newX <= size(in,1) && newY > 0 && newY <= size(in,2)
                out(x,y) = in(newX,newY);
            end

        end
    end

end

result = shearImage(clown, 1, 0);
imshow(result);
```

