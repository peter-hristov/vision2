% Show scatter plots of RGB values and HS (hue-saturation) values.
% Cluster on RGB/HS and label each pixel accordingly.
% Show segmented image by colouring each pixel with cluster-centroid
% (assume fixed V for HS).

% Image to experiment on
ifilename = 'clothworkers.jpg';

% Read image, convert to [0 1] range, and 'rasterize' into a row for each
% pixel with RGB/HS in columns
A = double(imread(ifilename))/255;
[r c t] = size(A);
X = reshape(A,r*c,t);

% RGB clustering
[idx C] = kmeans(X,3);

% Scatter plots of RGB pixel values
scatter3(X(:,1),X(:,2),X(:,3),3,X);      % original pixel colouring
xlabel('red'); ylabel('green'), zlabel('blue');
scatter3(X(:,1),X(:,2),X(:,3),2,idx);    % cluster-prototype colouring
xlabel('red'); ylabel('green'), zlabel('blue');
colormap(C);

% Reform image from rows of pixels and write to disk
D = reshape(C(idx,:),r,c,3);
imwrite(D,'RGB-segmented image Clothworkers.bmp','bmp');

% HS clustering
XX = rgb2hsv(X);    % convert to HSV
XX(:,2) = 0.5*XX(:,2);  % reduce variance of saturation (untidy)
[idx C] = kmeans(XX(:,1:2),3);  % cluster HS

% Convert cluster-centroid HS values to RGB by assuming fixed V.
CC = hsv2rgb(cat(2,C,[0.6 0.6 0.6]'));

% Scatter plots of HS pixel values
scatter(XX(:,1),XX(:,2),1,XX);
xlabel('hue'); ylabel('saturation');
scatter(XX(:,1),XX(:,2),1,idx);
xlabel('hue'); ylabel('saturation');
colormap(CC);

% Reform image from rows of pixels and write to disk
D = reshape(CC(idx,:),r,c,3);
imwrite(D,'HS-segmented image Clothworkers.bmp','bmp');





