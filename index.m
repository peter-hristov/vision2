% Read images
[foo Anames] = fileattrib( 'images/butterfly/image*');
[foo Cnames] = fileattrib( 'images/chairs/image*');
[foo Bnames] = fileattrib( 'images/laptop/image*');
[foo Dnames] = fileattrib( 'images/motorbikes/image*');


% Parameters
patchNum = 500;
patchSize = 5;
imageNum = 200;
codebookSize = 10;

images = cat(2, Anames, Bnames, Cnames, Dnames);

% Select 5 of each at random to be test images
testIndices = [randperm(50, 5); randperm(50, 5) + 50; randperm(50, 5) + 100; randperm(50, 5) + 150];

F = zeros(patchNum * imageNum, patchSize * patchSize);

% Read and filter images
for i = 1:imageNum

    image = im2double(imread(images(i).Name));

    % If image was read in an RGB format
    if size(image, 3) > 1
        image = rgb2gray(image);
    end

    % Filter image
    image = lcn(image);

    % Get feature coordinates (top left corner)
    patchesX = randi([1, size(image, 1) - patchSize], patchNum, 1);
    patchesY = randi([1, size(image, 2) - patchSize], patchNum, 1);
    P = [patchesX patchesY];

    images(i).Features = zeros(patchNum, patchSize^2);

    % Extract descriptors
    for j = 1:patchNum
        feature = image(P(j, 1):(P(j, 1) + patchSize - 1), P(j, 2):(P(j, 2) + patchSize - 1));
        images(i).Features(j, :) = feature(:);
        %imwrite(feature, ['./features/element-' num2str(i) '-' num2str(j) '.bmp'], 'bmp');
    end

end

% Get Codebook by clustering all features in the training data
[idx Codebook] = kmeans(vertcat(images(~ismember(1:end, testIndices(:))).Features), codebookSize);

% Write the codebook
for i = 1:codebookSize
    codeWord = reshape(Codebook(i, :)', patchSize, patchSize);
    imwrite(codeWord, ['./codebook/element-' num2str(i) '.bmp'], 'bmp');
end

% Make histograms for all images
for i = 1:imageNum
    A = nearestneighbour(images(i).Features, Codebook);
    images(i).Histogram = histcounts(A, codebookSize , 'Normalization', 'probability');
end


%f = F(1:5,:);
%size(f);
%size(C);

%F(1,:)
%A = nearestneighbour(f, C);

%size(C(A, :))

%A
%H = histcounts(A, 10, 'Normalization', 'probability')

%a = reshape(C(i, :)', patchSize, patchSize);
%b = reshape(C(i, :)', patchSize, patchSize);

%figure
%imshow(f)

%figure
%imshow(f)

%clear all




