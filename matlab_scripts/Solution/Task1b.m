close all; clear; clc;

%% Task 1b: Low-light image enhancement using nonlinear filtering
% 1) Read an RGB image.
% 2) Convert RGB -> HSV using rgb2hsv().
% 3) Take only V channel (brightness) V(x,y).
% 4) Compute U(x,y) by applying Task 1a nonlinear filtering to V:
%       U = NLFilter(V, k, N=20)
% 5) Compute enhanced brightness:
%       E(x,y) = V(x,y) / ( U(x,y)^p + r )
%       r = 0.01, p in [0.8, 0.95]
% 6) Replace V channel by E, convert HSV -> RGB using hsv2rgb().
% 7) Run on 3-4 images from DarkFace dataset and show results.

%% User settings
imagesDir = "../../images/DarkFace/image";   % DarkFace folder
numImagesToShow = 4;          % requires 3-4 images

% Task requires:
N_filter = 20;      % 20 iterations
r = 0.01;           % avoid division by zero

% try p between 0.8 and 0.95
p_list = [0.80, 0.85, 0.90, 0.95];

% k controls the nonlinearity strength in the filter.
k = 50;

%% Collect images from dataset
% We search recursively for jpg/png/jpeg images inside datasetDir.
imgFiles = [
    dir(fullfile(imagesDir, "*.jpg"));
    dir(fullfile(imagesDir, "*.png"));
    dir(fullfile(imagesDir, "*.jpeg"))
];

if isempty(imgFiles)
    error("No images found in: %s", imagesDir);
end

fprintf("Found %d images in folder: %s\n", length(imgFiles), imagesDir);

% Make sure numImagesToShow is not larger than total
numImagesToShow = min(numImagesToShow, length(imgFiles));

% Randomly select images
randIdx = randperm(length(imgFiles), numImagesToShow);
selectedFiles = imgFiles(randIdx);


fprintf("Found %d images in dataset.\n", length(imgFiles));
fprintf("Using %d images for experiments.\n", numImagesToShow);

%% Process each selected image
for idx = 1:numImagesToShow

    imgPath = fullfile(selectedFiles(idx).folder, selectedFiles(idx).name);
    fprintf("\nProcessing image %d/%d: %s\n", idx, numImagesToShow, imgPath);

    % Read RGB image
    I_rgb = im2double(imread(imgPath));

    % If some image is grayscale by accident, convert to 3-channel RGB
    if ndims(I_rgb) == 2
        I_rgb = repmat(I_rgb, [1 1 3]);
    end

    % Convert RGB -> HSV
    I_hsv = rgb2hsv(I_rgb);

    % Extract V channel (brightness)
    V = I_hsv(:,:,3);

    % Apply nonlinear filtering to V to get U
    % U is the filtered brightness (smoothed but edge-preserving)
    U = nonlinear_filter_3x3(V, k, N_filter);

    %For each gamma p, compute enhancement E and show results

    figure('Name', sprintf("Task 1b - Image %d: %s", idx, selectedFiles(idx).name), ...
           'NumberTitle', 'off');

    % Layout: original + 4 enhanced results (for different p)
    tiledlayout(1, 1 + length(p_list), 'Padding', 'compact', 'TileSpacing', 'compact');

    % ---- Tile 1: original ----
    nexttile;
    imshow(I_rgb, []);
    title(sprintf("Original\n%s", selectedFiles(idx).name), 'Interpreter', 'none');

    for pp = 1:length(p_list)
        p = p_list(pp);

        % Enhancement formula
        % E(x,y) = V(x,y) / ( U(x,y)^p + r )
        E = V ./ ( (U.^p) + r );

        % Important: keep values in [0,1] for valid HSV
        E = min(max(E, 0), 1);

        % Replace V channel by enhanced E
        I_hsv_enh = I_hsv;
        I_hsv_enh(:,:,3) = E;

        % Convert back to RGB
        I_rgb_enh = hsv2rgb(I_hsv_enh);

        % Show
        nexttile;
        imshow(I_rgb_enh, []);
        title(sprintf("Enhanced\np=%.2f", p));
    end

end

%% Local function: Nonlinear 3x3 iterative filter (Task 1a)
function out = nonlinear_filter_3x3(I, k, N)
% NONLINEAR_FILTER_3X3
% Implements the iterative nonlinear filtering scheme:
%   I0(x,y) = I(x,y)
%   In+1(x,y) = sum w_ij * In(x+i, y+j) / sum w_ij
%   w_ij = exp( -k * | In(x,y) - In(x+i, y+j) | )
% Input:
%   I : grayscale image in [0,1]
%   k : positive constant controlling weight decay
%   N : number of iterations
% Output:
%   out : filtered image after N iterations

    [rows, cols] = size(I);

    im1 = I;                 % current iteration image In
    im2 = zeros(rows, cols); % next iteration image In+1

    for iter = 1:N

        % To avoid black borders, copy previous image first.
        % Then update only inner pixels.
        im2 = im1;

        for x = 2:rows-1
            for y = 2:cols-1

                center = im1(x,y);

                weighted_sum = 0;
                weight_total = 0;

                % 3x3 neighborhood
                for i = -1:1
                    for j = -1:1

                        neighbor = im1(x+i, y+j);

                        % weight depends on local intensity difference
                        w = exp(-k * abs(center - neighbor));

                        weighted_sum = weighted_sum + w * neighbor;
                        weight_total = weight_total + w;

                    end
                end

                % normalized weighted average
                im2(x,y) = weighted_sum / weight_total;

            end
        end

        % update
        im1 = im2;
    end

    out = im1;
end
