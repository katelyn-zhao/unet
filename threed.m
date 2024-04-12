mask_directory = 'kunet/prediction slice only/';
mask_dataset = {};

% List all files in the mask directory
masks = dir(fullfile(mask_directory, '*.png'));  % Assuming all files are PNGs

% % % % Iterate over each file in the directory
for i = 1:numel(masks)
    % Check if the image name starts with '8394/8406'
    image_name_parts = strsplit(masks(i).name, '_');
    if strcmp(image_name_parts{2}(1:4), '8406')
        % Load the image
        image = imread(fullfile(mask_directory, masks(i).name));
        
        % Append the image to the mask dataset cell array
        mask_dataset{end+1} = image;
    end
end

% Specify the desired dimensions for the images
desired_rows = 800;
desired_cols = 800;

% Initialize a 3D numeric array to store the stacked images
volume = zeros(desired_rows, desired_cols, numel(mask_dataset));

% Stack 2D images in mask_dataset along the third dimension
for i = 1:numel(mask_dataset)
    % Resize the image to the desired dimensions
    resized_image = imresize(mask_dataset{i}, [desired_rows, desired_cols]);
    
    % Convert the resized image to grayscale
    grayscale_image = rgb2gray(resized_image);
    
    % Store the grayscale image in the volume
    volume(:, :, i) = grayscale_image;
end

% Generate an isosurface from the 3D volume
iso = isosurface(volume, 0.5);

% Plot the isosurface
p = patch(iso);
isonormals(volume, p);
p.FaceColor = 'red';
p.EdgeColor = 'none';
daspect([1 1 1]);
view(3);
axis tight;
camlight;
