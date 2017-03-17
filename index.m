[foo Anames] = fileattrib( 'images/butterfly/image*');
[foo Cnames] = fileattrib( 'images/chairs/image*');
[foo Bnames] = fileattrib( 'images/laptop/image*');
[foo Dnames] = fileattrib( 'images/motorbikes/image*');

images = cat(2,Anames,Bnames,Cnames,Dnames);


% Read and filter images
%for i = 1:size(images, 2)
for i = 1:10

    image = im2double(imread(images(i).Name));

    if size(image, 3) > 1
        image = rgb2gray(image);
    end

    image = lcn(image);

    images(i).Data = image;
end


clear all
