close all; clear; clc;

%% Task 2: Local histogram equalisation
% - Implement a local histogram equalization scheme
% - Apply to unevenly illuminated images (e.g., tun.jpg)
% - Evaluate suitability for low-light enhancement
% Notes:
% - We implement local HE on grayscale using a sliding window.
% - For color images, we process the V channel in HSV to avoid color shifts.

%% Settings
imgPath = "../../images/tun.jpg";   % tun.jpg path
% imgPath = "../../images/low_light/samsung_galaxy.jpg";
winSize = 31;                      % local window size (odd): 15/31/51 are typical the larger the closer it is to the global HE
nBins   = 256;                     % histogram bins (8-bit)

% For comparison baselines
doCompareWithGlobalHE = true;
doCompareWithCLAHE    = true;  % uses adapthisteq (allowed as comparison baseline)

%% Read image
I = im2double(imread(imgPath));
isColor = (ndims(I) == 3);

if ~isColor
    I_gray = I;
else
    % Use HSV: apply enhancement to V channel only (brightness)
    I_hsv = rgb2hsv(I);
    I_gray = I_hsv(:,:,3);  % treat V as grayscale target
end

%% (A) Local histogram equalization (our implementation)
fprintf("Running local histogram equalization (win=%d, bins=%d)...\n", winSize, nBins);
I_local = local_hist_equalize(I_gray, winSize, nBins);

%% (B) Global histogram equalization (baseline)
if doCompareWithGlobalHE
    % histeq expects [0,1] grayscale; returns [0,1]
    I_global = histeq(I_gray, nBins);
end

%% (C) CLAHE (baseline) - Matlab adapthisteq
if doCompareWithCLAHE
    % CLAHE is a common improved local HE (contrast-limited)
    I_clahe = adapthisteq(I_gray, ...
        'NumTiles', [8 8], ...      % grid of contextual regions
        'ClipLimit', 0.01);         % contrast limiting
end

%% Reconstruct output images
if ~isColor
    out_local  = I_local;
    out_global = [];
    out_clahe  = [];
    if doCompareWithGlobalHE, out_global = I_global; end
    if doCompareWithCLAHE,    out_clahe  = I_clahe;  end
else
    % Replace V channel and convert back to RGB for display
    out_local = I_hsv;
    out_local(:,:,3) = I_local;
    out_local = hsv2rgb(out_local);

    if doCompareWithGlobalHE
        out_global = I_hsv; out_global(:,:,3) = I_global; out_global = hsv2rgb(out_global);
    else
        out_global = [];
    end

    if doCompareWithCLAHE
        out_clahe = I_hsv; out_clahe(:,:,3) = I_clahe; out_clahe = hsv2rgb(out_clahe);
    else
        out_clahe = [];
    end
end

%% Visualization: results + histograms
figure('Name', 'Task 2: Local Histogram Equalisation', 'NumberTitle', 'off');

if doCompareWithGlobalHE && doCompareWithCLAHE
    tiledlayout(2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    nexttile; imshow(I, []);          title("Original");
    nexttile; imshow(out_global, []); title("Global HE (histeq)");
    nexttile; imshow(out_local, []);  title(sprintf("Local HE (win=%d)", winSize));
    nexttile; imshow(out_clahe, []);  title("CLAHE (adapthisteq)");
elseif doCompareWithGlobalHE
    tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile; imshow(I, []);          title("Original");
    nexttile; imshow(out_global, []); title("Global HE (histeq)");
    nexttile; imshow(out_local, []);  title(sprintf("Local HE (win=%d)", winSize));
elseif doCompareWithCLAHE
    tiledlayout(1, 3, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile; imshow(I, []);         title("Original");
    nexttile; imshow(out_local, []); title(sprintf("Local HE (win=%d)", winSize));
    nexttile; imshow(out_clahe, []); title("CLAHE (adapthisteq)");
else
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    nexttile; imshow(I, []);         title("Original");
    nexttile; imshow(out_local, []); title(sprintf("Local HE (win=%d)", winSize));
end

% Optional: show histograms of the processed channel
figure('Name', 'Histograms on processed channel', 'NumberTitle', 'off');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% --- Original histogram ---
nexttile;
histogram(I_gray(:), nBins);
xlim([0 1]);
title("Original channel histogram");
xlabel("Intensity");
ylabel("Count");

% --- Local HE histogram ---
nexttile;
histogram(I_local(:), nBins);
xlim([0 1]);
title("Local HE channel histogram");
xlabel("Intensity");
ylabel("Count");

%% Simple evaluation notes printed to console
fprintf("\n================= Task 2 Evaluation Notes =================\n");
fprintf("- Local HE can correct uneven illumination by adapting contrast locally.\n");
fprintf("- However, plain Local HE often amplifies noise in very dark regions.\n");
fprintf("- CLAHE usually reduces this problem via contrast limiting.\n");
fprintf("Try different winSize (e.g., 15/31/51) to see trade-offs.\n");
fprintf("===========================================================\n");

%% Local function: Local histogram equalization (our implementation)
function out = local_hist_equalize(I, winSize, nBins)
% LOCAL_HIST_EQUALIZE
% Basic local histogram equalization:
% For each pixel, build histogram in a local window and map the center pixel
% using the local CDF.
% Input:
%   I       : grayscale image in [0,1]
%   winSize : odd integer window size (e.g., 31)
%   nBins   : number of histogram bins (e.g., 256)
% Output:
%   out     : locally histogram-equalized image in [0,1]

    if mod(winSize, 2) == 0
        error("winSize must be odd.");
    end

    [rows, cols] = size(I);
    pad = floor(winSize / 2);

    % Pad image to handle borders (replicate avoids artificial dark borders)
    Ipad = padarray(I, [pad pad], 'replicate', 'both');

    out = zeros(rows, cols);

    % Precompute bin edges for discretization
    % Map [0,1] -> 1..nBins
    for x = 1:rows
        for y = 1:cols

            % Extract local window
            wx = x : x + 2*pad;
            wy = y : y + 2*pad;
            patch = Ipad(wx, wy);

            % Compute histogram (integer bin indices)
            % Convert patch values to bin indices in [1..nBins]
            idx = floor(patch * (nBins - 1)) + 1;

            h = zeros(nBins, 1);
            % Accumulate histogram counts
            for t = 1:numel(idx)
                h(idx(t)) = h(idx(t)) + 1;
            end

            % Compute CDF
            cdf = cumsum(h) / sum(h);

            % Map center pixel using local CDF
            centerVal = I(x, y);
            centerBin = floor(centerVal * (nBins - 1)) + 1;

            out(x, y) = cdf(centerBin);

        end
    end

    % Ensure valid range
    out = min(max(out, 0), 1);
end
