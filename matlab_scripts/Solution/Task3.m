close all; clear; clc;

%% Task 3: Image filtering in frequency domain (Notch filtering)
% Goal:
% - Compute and visualise Fourier transform using:
%       log(abs(fftshift(fft2(...))))
% - Identify periodic noise frequencies (4 small crosses) using impixelinfo
% - Construct a notch filter (band-stop filter) using small circles/rectangles
% - Suppress periodic noise while preserving image quality
% - Show:
%   (1) corrupted image
%   (2) Fourier magnitude spectrum
%   (3) notch filter mask in frequency domain
%   (4) reconstructed image after filtering

%% Read image
imgPath = "../../images/eye-hand.png";

I_rgb = imread(imgPath);
I = im2double(im2gray(I_rgb));   % grayscale double in [0,1]

[rows, cols] = size(I);

%% Fourier transform
F = fft2(I);
F_shift = fftshift(F);

% magnitude spectrum for visualization
mag = log(1 + abs(F_shift));

%% Show image + spectrum (use impixelinfo to locate noise peaks)
figure('Name', 'Task 3: Locate periodic noise peaks', 'NumberTitle', 'off');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imshow(I, []);
title("Corrupted image (eye-hand.png)");

nexttile;
imshow(mag, []);
title("Magnitude spectrum: log(1 + |fftshift(F)|)");

% Use impixelinfo to read coordinates of bright peaks.
impixelinfo;

fprintf("\n============================================================\n");
fprintf("Use impixelinfo on the spectrum figure to find the 4 bright peaks.\n");
fprintf("Write down their (x,y) coordinates in the shifted spectrum image.\n");
fprintf("The center is approximately (cols/2, rows/2).\n");
fprintf("============================================================\n");

%% Step 2: Construct notch filter
% fill in peak coordinates after reading them from impixelinfo.
% NOTE:
% - Coordinates in Matlab display are (x,y) = (column, row).
% - F_shift has size rows x cols.

peaks = [
    128, 157;
    385, 100;
    128, 413;
    385, 357
];

%%  Notch filter parameters
notchRadius = 8;     % radius of each notch (try 3~10)
mask = ones(rows, cols);   % 1 = keep frequency, 0 = remove

% Build circular notches at peak locations
for t = 1:size(peaks, 1)

    x0 = peaks(t, 1);   % column
    y0 = peaks(t, 2);   % row

    if x0 == 0 && y0 == 0
        continue; % skip placeholder rows
    end

    for y = 1:rows
        for x = 1:cols
            if (x - x0)^2 + (y - y0)^2 <= notchRadius^2
                mask(y, x) = 0;
            end
        end
    end
end

%% Apply notch filter in frequency domain
F_filtered_shift = F_shift .* mask;

% Inverse shift and inverse FFT
F_filtered = ifftshift(F_filtered_shift);
I_rec = real(ifft2(F_filtered));

% Clip for display
I_rec = min(max(I_rec, 0), 1);

%% Visualisation: mask + filtered spectrum + reconstructed image
figure('Name', 'Task 3: Notch filtering results', 'NumberTitle', 'off');
tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imshow(I, []);
title("Original corrupted image");

nexttile;
imshow(mag, []);
title("Original spectrum log(1+|F|)");

nexttile;
imshow(mask, []);
title(sprintf("Notch filter mask (radius=%d)", notchRadius));

nexttile;
imshow(I_rec, []);
title("Reconstructed image after notch filtering");

%% Optional: show filtered spectrum
mag_filtered = log(1 + abs(F_filtered_shift));

figure('Name', 'Filtered spectrum', 'NumberTitle', 'off');
imshow(mag_filtered, []);
title("Filtered spectrum log(1+|F_filtered|)");
