
[foo Anames] = fileattrib( 'images/butterfly/image*');
[foo Cnames] = fileattrib( 'images/chairs/image*');
[foo Bnames] = fileattrib( 'images/laptop/image*');
[foo Dnames] = fileattrib( 'images/motorbikes/image*');

images = cat(2,Anames,Bnames,Cnames,Dnames);
%images = Anames;

patchNum = 100;
patchSize = 100;
imageNum = 10;
codebookSize = 10;

F = [];

% Read and filter images
%for i = 1:size(images, 2)
for i = 1:imageNum

    image = im2double(imread(images(i).Name));

    if size(image, 3) > 1
        image = rgb2gray(image);
    end

    image = lcn(image);


    patchesX = randi([1, size(image, 1) - patchSize], patchNum, 1);
    patchesY = randi([1, size(image, 2) - patchSize], patchNum, 1);
    P = [patchesX patchesY];

    images(i).Features = [];

    for j = 1:size(P, 1)
        feature = image(P(j, 1):(P(j, 1) + patchSize - 1), P(j, 2):(P(j, 2) + patchSize - 1));
        images(i).Features = [images(i).Features feature(:)];

        F = [F feature(:)];

        %figure
        %imshow(feature)
    end

end

% Get Code Words by clustering
[idx C] = kmeans(F', codebookSize);  % cluster HS

% Print the codebook
for i = 1:size(C, 1)
    x = C(i, :)';
    x = reshape(x, patchSize, patchSize);

    %figure
    %imshow(x)

    imwrite(x,['./codebook/element-' num2str(i)], 'bmp');

end


clear all
