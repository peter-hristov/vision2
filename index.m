for i = 1:1
    % Read images
    [foo Anames] = fileattrib( 'images/butterfly/image*');
    [foo Cnames] = fileattrib( 'images/chairs/image*');
    [foo Bnames] = fileattrib( 'images/laptop/image*');
    [foo Dnames] = fileattrib( 'images/motorbikes/image*');

    % Parameters Default - 1000, 15, 50, 5, 150, 0, 0 For 50-70% Accuracy
    patchNum = 50;
    patchSize = 15;
    imageNum = 10;
    testNum = 2;
    codebookSize = 5;

    tree = 1;
    binaryHistogram = 0;
    normalizeBinaryHistogram = 0;

    % Get a subset of the images for faster processing
    images = cat(2, Anames(1:imageNum), Bnames(1:imageNum), Cnames(1:imageNum), Dnames(1:imageNum));

    % Select 5 of each at random to be test images
    testIndices = [randperm(imageNum, testNum); randperm(imageNum, testNum) + 1 * imageNum; randperm(imageNum, testNum) + 2 * imageNum; randperm(imageNum, testNum) + 3 * imageNum];

    testIndices

    tic;

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

        if images(i).Class == 1
            images(i).Category = 'b';
        end
        if images(i).Class == 2
            images(i).Category = 'l';
        end
        if images(i).Class == 3
            images(i).Category = 'c';
        end
        if images(i).Class == 4
            images(i).Category = 'p';
        end
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

    if binaryHistogram > 0
        images(i).Histogram(images(i).Histogram > 0) = [1];
        if normalizeBinaryHistogram > 0
            images(i).Histogram = histcounts(images(i).Histogram, codebookSize , 'Normalization', 'probability');
        end
    end
end

% Get which nearest neighbour for he histograms
TestImages     = images( ismember(1:end, testIndices(:)));
TrainingImages = images(~ismember(1:end, testIndices(:)));

if tree > 0
    % Decision Tree
    %Mdl = fitctree(vertcat(TrainingImages.Histogram), num2str(vertcat(TrainingImages.Class)));

    X = vertcat(TrainingImages.Histogram);
    %Y = double(vertcat(TrainingImages.Class));

    Y = vertcat(TrainingImages.Category);

    [X Y]

    Mdl = fitcnb(X, Y);

    Closest = predict(Mdl, vertcat(TestImages.Histogram));

    G = [[TestImages.Class] ; str2num(Closest)'];

else
    % Nearest Neighbour
    Closest = nearestneighbour(vertcat(TestImages.Histogram), vertcat(TrainingImages.Histogram));
    G = [TestImages.Class ; TrainingImages(Closest).Class];
end


% Get the number of times we got things wrong
m = 0;
e = zeros(4, 4);

for i = 1:size(G, 2)

    if (G(1, i) == G(2, i))
        m = m + 1;
    end

    e(G(1, i), G(2, i)) = e(G(1, i), G(2, i)) + 1;
end

disp(['Correct : ' num2str(m) '/' num2str(size(G,2)) ' = ' num2str(100 * m/size(G,2)) '%'])
e

toc;

clear all

end


