close all; clear; clc;

%% Task 4a: Image deblurring by the Wiener filter
% Model:
%   g(x,y) = h(x,y) ⊗ f(x,y) + n(x,y)
% Fourier domain:
%   G(u,v) = H(u,v)F(u,v) + N(u,v)
% Wiener filter solution:
%   F_hat(u,v) = ( conj(H(u,v)) / ( |H(u,v)|^2 + K ) ) * G(u,v)
% Requirements:
% - Do NOT use deconvwnr
% - Use psf2otf to handle blur kernel
% - Test on motion blur and Gaussian blur with different kernels

%% 1) Read test image (grayscale)
imgPath = "../../images/cameraman.tif";     % you can change to barbara_face.png etc.
f = im2double(imread(imgPath));

if ndims(f) == 3
    f = rgb2gray(f);
end

%%  2) Noise settings
noise_mean = 0;
noise_var  = 1e-6;   % 1e-6 or 1e-5 are typical

%% 3) Experiment list (Gaussian + Motion)
% We will test multiple kernels to satisfy rubric.

expList = {};

% --- Gaussian blur kernels ---
expList{end+1} = struct("type","gaussian", "size",[9 9],  "sigma",2);
expList{end+1} = struct("type","gaussian", "size",[15 15],"sigma",4);

% --- Motion blur kernels ---
expList{end+1} = struct("type","motion", "len",15, "theta",0);
expList{end+1} = struct("type","motion", "len",25, "theta",45);

%%  4) Wiener K values (try several)
% K approximates noise-to-signal power ratio.
K_list = [1e-4, 5e-4, 1e-3];

%% 5) Run experiments

for e = 1:length(expList)

    %% Create blur kernel h(x,y)
    exp = expList{e};

    if exp.type == "gaussian"
        h = fspecial("gaussian", exp.size, exp.sigma);
        expName = sprintf("Gaussian (%dx%d, sigma=%.1f)", exp.size(1), exp.size(2), exp.sigma);

    elseif exp.type == "motion"
        h = fspecial("motion", exp.len, exp.theta);
        expName = sprintf("Motion (len=%d, theta=%d)", exp.len, exp.theta);
    end

    % Normalize kernel to ensure sum(h)=1
    h = h / sum(h(:));

    %% Blur + add noise to generate degraded image g(x,y)
    g = imfilter(f, h, "conv", "circular");
    g = imnoise(g, "gaussian", noise_mean, noise_var);

    %% Convert blur kernel to OTF (frequency response)
    % H(u,v) has the same size as image g
    H = psf2otf(h, size(g));

    %% FFT of degraded image
    G = fft2(g);

    %% Display results for different K values
    figure("Name", sprintf("Task 4a: %s", expName), "NumberTitle","off");
    tiledlayout(1, 2 + length(K_list), "Padding","compact","TileSpacing","compact");

    % ---- Tile 1: original ----
    nexttile;
    imshow(f, []);
    title("Original f(x,y)");

    % ---- Tile 2: blurred + noisy ----
    nexttile;
    imshow(g, []);
    title(sprintf("Blurred+Noise g(x,y)\nPSNR=%.2f dB", psnr(g,f)));

    %% Wiener restoration for each K
    for kk = 1:length(K_list)

        K = K_list(kk);

        % Wiener filter:
        % W(u,v) = conj(H) / (|H|^2 + K)
        W = conj(H) ./ (abs(H).^2 + K);

        % Apply Wiener filter in frequency domain
        F_hat = W .* G;

        % Inverse FFT to reconstruct
        f_rec = real(ifft2(F_hat));

        % Clip to valid range [0,1]
        f_rec = min(max(f_rec, 0), 1);

        % Show reconstructed image
        nexttile;
        imshow(f_rec, []);
        title(sprintf("Wiener\nK=%.0e\nPSNR=%.2f", K, psnr(f_rec,f)));
    end

end
