for i = 1:1

    % Read images
    [foo Anames] = fileattrib( 'images/butterfly/image*');
    [foo Cnames] = fileattrib( 'images/chairs/image*');
    [foo Bnames] = fileattrib( 'images/laptop/image*');
    [foo Dnames] = fileattrib( 'images/motorbikes/image*');

    patchNums = [100 250 500 1000 1500];
    codebookSizes = [5 10 50 100 200];

    % Parameters Default - 1000, 15, 50, 5, 150, 0, 0 For 50-70% Accuracy
    patchNum = 500;
    patchSize = 20;
    imageNum = 10;
    testNum = 2;
    codebookSize = 36;

    %patchNum = 1000;
    %codebookSize = 100;


    % Neighbour 0, Tree 1, Bayes 2
    tree = 0;
    binaryHistogram = 0;
    normalizeBinaryHistogram = 0;

    Parameters = [patchNum patchSize imageNum testNum codebookSize tree binaryHistogram normalizeBinaryHistogram]

    % Get a subset of the images for faster processing
    images = cat(2, Anames(1:imageNum), Bnames(1:imageNum), Cnames(1:imageNum), Dnames(1:imageNum));

    % Select 5 of each at random to be test images
    testIndices = [randperm(imageNum, testNum); randperm(imageNum, testNum) + 1 * imageNum; randperm(imageNum, testNum) + 2 * imageNum; randperm(imageNum, testNum) + 3 * imageNum];

    %testIndices = [48    14    24    33    30 55    94    78    93    97 106   147   136   144   127 154   157   176   175   198];

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

            if abs(sum(feature(:))) > 10
                images(i).Features(j, :) = feature(:);
                [i abs(sum(feature(:)))]
            else
               abs(sum(feature(:)))
            end

            %imwrite(feature, ['./features/element-' num2str(i) '-' num2str(j) '.bmp'], 'bmp');
        end

        images(i).Class = idivide(int32(i-1), int32(imageNum)) + 1;

    end

    % Get Codebook by clustering all features in the training data (i.e. not in the test data)
    [idx Codebook] = kmeans(vertcat(images(~ismember(1:end, testIndices(:))).Features), codebookSize);


    size(Codebook)

    % Write the codebook
    for i = 1:codebookSize
        codeWord = reshape(Codebook(i, :)', patchSize, patchSize);
        imwrite(codeWord, ['./codebook/element-' num2str(i) '.bmp'], 'bmp');

        [i sum(Codebook(i, :)')]

        subplot(6, 6, i);
        codeWord = reshape(Codebook(i, :)', patchSize, patchSize);
        plot(codeWord);
        imshow(codeWord);
    end

    figure


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


    %subplot(2, 2, 1);
    %codeWord = reshape(images(1).Features(1, :)', patchSize, patchSize);
    %imshow(codeWord);

    %subplot(2, 2, 2);
    %codeWord = reshape(images(1).Features(2, :)', patchSize, patchSize);
    %imshow(codeWord);

    %subplot(2, 2, 3);
    %codeWord = reshape(images(1).Features(3, :)', patchSize, patchSize);
    %imshow(codeWord);

    %subplot(2, 2, 4);
    %codeWord = reshape(images(1).Features(4, :)', patchSize, patchSize);
    %imshow(codeWord);


    subplot(2, 2, 1);
    plot(images(1).Histogram);

    subplot(2, 2, 2);
    plot(images(12).Histogram);

    subplot(2, 2, 3);
    plot(images(25).Histogram);

    subplot(2, 2, 4);
    plot(images(39).Histogram);

    % Get which nearest neighbour for he histograms
    TestImages     = images( ismember(1:end, testIndices(:)));
    TrainingImages = images(~ismember(1:end, testIndices(:)));

    if tree == 0
        % Neighbour
        Closest = nearestneighbour(vertcat(TestImages.Histogram), vertcat(TrainingImages.Histogram));
        G = [TestImages.Class ; TrainingImages(Closest).Class];
        %T = vertcat(TrainingImages(Closest).Name)
    end
    if tree == 1
        % Tree
        X = vertcat(TrainingImages.Histogram);
        Y = num2str(vertcat(TrainingImages.Class));

        Mdl = fitctree(vertcat(TrainingImages.Histogram), num2str(vertcat(TrainingImages.Class)));

        Closest = predict(Mdl, vertcat(TestImages.Histogram));

        G = [[TestImages.Class] ; str2num(Closest)'];
    end
    if tree == 2
        % Bayes
        X = vertcat(TrainingImages.Histogram);
        Y = num2str(vertcat(TrainingImages.Class));

        %Mdl = fitctree(vertcat(TrainingImages.Histogram), num2str(vertcat(TrainingImages.Class)));
        Mdl = fitcnb(X, Y, 'DistributionNames','kernel');

        Closest = predict(Mdl, vertcat(TestImages.Histogram));

        G = [[TestImages.Class] ; str2num(Closest)'];

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

    %clear all
end

