close all; clear; clc;
% matlab install Computer Vision Toolbox

%% Task 1c: Face detection on dark images vs enhanced images (DarkFace)
% - Use a publicly available face detection program (Matlab).
% - Compare detection results on:
%   (1) original low-light images
%   (2) images enhanced by Task 1b
% - Conclude whether enhancement improves detection results.
% Detector:
% - Matlab Computer Vision Toolbox:
%   vision.CascadeObjectDetector (publicly available)
% Enhancement:
% - Task 1b (HSV-based enhancement using nonlinear filtering on V channel)

%% Settings
imagesDir = "../../images/DarkFace/image";  
K = 10;                                    % number of random images

% Task 1b parameters
N_filter = 20;     % number of iterations for nonlinear filtering (Task 1a)
r = 0.01;          % avoid division by zero
p = 0.80;          % gamma parameter (choose in [0.8, 0.95])
k = 50;            % nonlinearity strength for Task 1a filter


%% Collect images
imgFiles = [
    dir(fullfile(imagesDir, "*.jpg"));
    dir(fullfile(imagesDir, "*.png"));
    dir(fullfile(imagesDir, "*.jpeg"))
];

if isempty(imgFiles)
    error("No images found in: %s", imagesDir);
end

fprintf("Found %d images in folder: %s\n", length(imgFiles), imagesDir);

% Manual selection (priority)
% If you put filenames here, we will use them first.
% If this list is empty, we will randomly select K images.
manualList = {
    "1.png"
    "123.png"
    "1081.png"
    "1732.png"
    "1784.png"
    "2410.png"
    "2422.png"
    "3179.png"
    "3320.png"
    "3637.png"
};

% Build selectedFiles
if ~isempty(manualList)

    fprintf("Manual selection list is NOT empty. Using manually selected images.\n");

    % Convert to struct array with fields {folder, name} like dir() output
    selectedFiles = struct('folder', {}, 'name', {});

    for m = 1:length(manualList)
        fname = manualList{m};
        fpath = fullfile(imagesDir, fname);

        if exist(fpath, 'file')
            selectedFiles(end+1).folder = imagesDir; %#ok<SAGROW>
            selectedFiles(end).name = fname;
        else
            warning("Manual file not found: %s", fpath);
        end
    end

    if isempty(selectedFiles)
        error("Manual list was given, but none of the files exist. Please check filenames.");
    end

    % Override K to match manual selection count
    K = length(selectedFiles);
    fprintf("Using %d manually selected images.\n", K);

else

    fprintf("Manual selection list is empty. Randomly selecting K=%d images.\n", K);

    K = min(K, length(imgFiles));
    randIdx = randperm(length(imgFiles), K);
    selectedFiles = imgFiles(randIdx);

    fprintf("Randomly selected %d images for Task 1c.\n", K);

end


%% Initialize face detector (Matlab public program)
detector = vision.CascadeObjectDetector();  
% default is frontal face detector

%%  Store statistics for report table
filename_list = strings(K, 1);
dark_faces = zeros(K, 1);
enh_faces  = zeros(K, 1);
improved   = strings(K, 1);

%%  Main loop over images
for idx = 1:K

    fileName = selectedFiles(idx).name;
    imgPath  = fullfile(selectedFiles(idx).folder, fileName);

    fprintf("\n[%d/%d] Processing: %s\n", idx, K, fileName);

    %Read RGB image
    I_dark = im2double(imread(imgPath));

    % Ensure it is RGB
    if ndims(I_dark) == 2
        I_dark = repmat(I_dark, [1 1 3]);
    end

    %% (A) Face detection on original dark image
    bboxes_dark = step(detector, I_dark);
    num_dark = size(bboxes_dark, 1);

    %% (B) Enhance image using Task 1b
    I_enh = enhance_lowlight_task1b(I_dark, k, N_filter, p, r);

    %% (C) Face detection on enhanced image
    bboxes_enh = step(detector, I_enh);
    num_enh = size(bboxes_enh, 1);

    %% Save stats
    filename_list(idx) = fileName;
    dark_faces(idx) = num_dark;
    enh_faces(idx)  = num_enh;

    if num_enh > num_dark
        improved(idx) = "Yes";
    elseif num_enh == num_dark
        improved(idx) = "Same";
    else
        improved(idx) = "No";
    end

    %% Visualization: side-by-side comparison with bounding boxes
    figure('Name', sprintf("Task 1c: %s", fileName), 'NumberTitle', 'off');

    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % ---- Left: original dark + boxes ----
    nexttile;
    imshow(I_dark, []);
    title(sprintf("Original (dark)\n%s\nFaces=%d", fileName, num_dark), ...
          'Interpreter', 'none');

    hold on;
    for b = 1:num_dark
        rectangle('Position', bboxes_dark(b,:), 'EdgeColor', 'g', 'LineWidth', 2);
    end
    hold off;

    % ---- Right: enhanced + boxes ----
    nexttile;
    imshow(I_enh, []);
    title(sprintf("Enhanced (Task1b)\np=%.2f, k=%d, N=%d\nFaces=%d", ...
          p, k, N_filter, num_enh));

    hold on;
    for b = 1:num_enh
        rectangle('Position', bboxes_enh(b,:), 'EdgeColor', 'g', 'LineWidth', 2);
    end
    hold off;

end

%% Summary table + quantitative conclusion
T = table(filename_list, dark_faces, enh_faces, improved, ...
          'VariableNames', {'filename', 'dark_num_faces', 'enhanced_num_faces', 'improved'});

disp("====================================================");
disp("Task 1c summary table:");
disp(T);

% Hit rate: percentage of images where at least one face is detected
dark_hit_rate = mean(dark_faces > 0);
enh_hit_rate  = mean(enh_faces  > 0);

% Average number of detections per image
dark_avg_faces = mean(dark_faces);
enh_avg_faces  = mean(enh_faces);

fprintf("\n================ Task 1c Quantitative Summary ================\n");
fprintf("Dark images:     hit rate = %.2f%%, avg faces = %.2f\n", 100*dark_hit_rate, dark_avg_faces);
fprintf("Enhanced images: hit rate = %.2f%%, avg faces = %.2f\n", 100*enh_hit_rate,  enh_avg_faces);

if enh_hit_rate > dark_hit_rate
    fprintf("Conclusion: Enhancement improves face detection hit rate.\n");
elseif enh_hit_rate == dark_hit_rate
    fprintf("Conclusion: Enhancement does not change the face detection hit rate.\n");
else
    fprintf("Conclusion: Enhancement reduces face detection hit rate (possible over-enhancement artifacts).\n");
end
fprintf("==============================================================\n");

%% Local function: Task 1b enhancement (HSV + nonlinear filter)
function I_rgb_enh = enhance_lowlight_task1b(I_rgb, k, N_filter, p, r)
% ENHANCE_LOWLIGHT_TASK1B
% Implements Task 1b enhancement:
% 1) RGB -> HSV
% 2) Take V channel
% 3) U = nonlinear_filter_3x3(V, k, N_filter)
% 4) E = V / (U^p + r)
% 5) Replace V by E, HSV -> RGB

    I_hsv = rgb2hsv(I_rgb);
    V = I_hsv(:,:,3);

    % Task 1a nonlinear filtering on V
    U = nonlinear_filter_3x3(V, k, N_filter);

    % Enhancement formula (Task 1b)
    E = V ./ ((U.^p) + r);

    % Clamp to [0,1] to keep HSV valid
    E = min(max(E, 0), 1);

    % Replace V channel and convert back
    I_hsv(:,:,3) = E;
    I_rgb_enh = hsv2rgb(I_hsv);
end

%% Local function: Task 1a nonlinear filter (3x3, iterative)
function out = nonlinear_filter_3x3(I, k, N)
% NONLINEAR_FILTER_3X3
% Implements Task 1a iterative nonlinear filtering:
%   I_{n+1}(x,y) = sum w_ij * I_n(x+i,y+j) / sum w_ij
%   w_ij = exp( -k * |I_n(x,y) - I_n(x+i,y+j)| )

    [rows, cols] = size(I);

    im1 = I;
    im2 = zeros(rows, cols);

    for iter = 1:N

        % avoid black border: copy previous image first
        im2 = im1;

        for x = 2:rows-1
            for y = 2:cols-1

                center = im1(x,y);

                weighted_sum = 0;
                weight_total = 0;

                for i = -1:1
                    for j = -1:1

                        neighbor = im1(x+i, y+j);

                        w = exp(-k * abs(center - neighbor));

                        weighted_sum = weighted_sum + w * neighbor;
                        weight_total = weight_total + w;

                    end
                end

                im2(x,y) = weighted_sum / weight_total;

            end
        end

        im1 = im2;
    end

    out = im1;
end
