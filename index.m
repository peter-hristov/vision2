% Read images
[foo Anames] = fileattrib( 'images/butterfly/image*');
[foo Cnames] = fileattrib( 'images/chairs/image*');
[foo Bnames] = fileattrib( 'images/laptop/image*');
[foo Dnames] = fileattrib( 'images/motorbikes/image*');

% Parameters
patchNum = 500;
patchSize = 15;
imageNum = 50;
testNum = 10;
codebookSize = 200;

% Get a subset of the images for faster processing
images = cat(2, Anames(1:imageNum), Bnames(1:imageNum), Cnames(1:imageNum), Dnames(1:imageNum));

% Select 5 of each at random to be test images
testIndices = [randperm(imageNum, testNum); randperm(imageNum, testNum) + 1 * imageNum; randperm(imageNum, testNum) + 2 * imageNum; randperm(imageNum, testNum) + 3 * imageNum];

testIndices

% Read and filter images
for i = 1:size(images, 2)

    % Read image
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

    images(i).Class = idivide(int32(i-1), int32(imageNum)) + 1;
    images(i).Index = i;
end

% Get Codebook by clustering all features in the training data (i.e. not in the test data)
[idx Codebook] = kmeans(vertcat(images(~ismember(1:end, testIndices(:))).Features), codebookSize);

% Write the codebook
for i = 1:codebookSize
    codeWord = reshape(Codebook(i, :)', patchSize, patchSize);
    imwrite(codeWord, ['./codebook/element-' num2str(i) '.bmp'], 'bmp');
end

% Make histograms for all images
for i = 1:size(images, 2)
    A = nearestneighbour(images(i).Features, Codebook);
    images(i).Histogram = histcounts(A, codebookSize , 'Normalization', 'probability');
    images(i).Histogram(images(i).Histogram > 0) = [1];

    images(i).Histogram = histcounts(images(i).Histogram, codebookSize , 'Normalization', 'probability');

end

% Get which nearest neighbour for he histograms
TestImages     = images( ismember(1:end, testIndices(:)));
TrainingImages = images(~ismember(1:end, testIndices(:)));

Closest = nearestneighbour(vertcat(TestImages.Histogram), vertcat(TrainingImages.Histogram));

% Get the number of times we got things wrong
G = [TestImages.Class ; TrainingImages(Closest).Class];
m = 0;
e = zeros(1, 4);

e


for i = 1:size(G, 2)
    if (G(1, i) ~= G(2, i))
        m = m + 1;
        e(G(1, i)) = e(G(1, i)) + 1;
    end
end

disp(['Correct : ' num2str(m) '/' num2str(size(G,2)) ' = ' num2str(100 * m/size(G,2)) '%'])
e

clear all

