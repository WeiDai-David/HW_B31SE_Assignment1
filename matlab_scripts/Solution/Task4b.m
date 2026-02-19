close all; clear; clc;

%% Task 4b: Image deblurring by ISRA
% Compare ISRA with:
% - Wiener (custom Wiener from Task4a)
% - Landweber
% - Richardson-Lucy
% Degradation model:
%   g = h ⊗ f + n
% ISRA iteration:
%   I0 = g
%   I_{n+1} = I_n .* [ (h(-x,-y) ⊗ g) ./ (h(-x,-y) ⊗ (h ⊗ I_n)) ]
% Notes:
% - In this task we keep the same settings as deblur.m
% - Because noise is additive Gaussian, RL and ISRA may not outperform Wiener.

%% 1) Choose blur kernel type: Gaussian OR Motion
testCase = "gaussian";   % "gaussian" or "motion"
% testCase = "motion";

%% 2) Load image f (ground truth)
f = im2double(imread("cameraman.tif"));
if ndims(f) == 3
    f = rgb2gray(f);
end

%% 3) Create blur kernel h
if testCase == "gaussian"
    h = fspecial("gaussian", [9 9], 4);
    caseName = "Gaussian blur (9x9, sigma=4)";
elseif testCase == "motion"
    h = fspecial("motion", 25, 45);
    caseName = "Motion blur (len=25, theta=45)";
end

h = h ./ sum(h(:));

%% 4) Blur operator (circular convolution)
blur = @(im) imfilter(im, h, "conv", "circular");

%% 5) Add Gaussian noise
noise_mean = 0;
noise_var  = 1e-6;

g = blur(f);
g = imnoise(g, "gaussian", noise_mean, noise_var);

%% 6) Frequency response H for fast computations
H = psf2otf(h, size(g));
G = fft2(g);

%% 7) Wiener baseline 
K_wiener = 1e-4;

% --- Custom Wiener (recommended, consistent with Task 4a) ---
W = conj(H) ./ (abs(H).^2 + K_wiener);
Fhat = W .* G;
WienerRec = real(ifft2(Fhat));
WienerRec = min(max(WienerRec, 0), 1);

%% 8) Initialize iterative methods
maxiter = 200;   % 3000 is too slow; 200 is enough for curves

RL = g;
Lw = g;
ISRA = g;

psnr0 = psnr(f, g);

psnrW  = psnr(WienerRec, f) * ones(maxiter, 1);
psnrRL = zeros(maxiter, 1);
psnrLw = zeros(maxiter, 1);
psnrIS = zeros(maxiter, 1);

%% 9) Precompute h(-x,-y) in frequency domain
% In circular convolution:
% h(-x,-y) corresponds to conj(H) in frequency domain.

%% 10) Iterations: RL, Landweber, ISRA
fprintf("Running iterative methods for: %s\n", caseName);
fprintf("Initial PSNR (blurred+noise): %.2f dB\n", psnr0);

for i = 1:maxiter

    %% ---------- Richardson-Lucy ----------
    % RL_{n+1} = RL_n .* [ h(-x) ⊗ ( g ./ (RL_n ⊗ h) ) ]
    denom_RL = blur(RL);
    denom_RL = max(denom_RL, 1e-12);   % avoid division by zero

    ratio = g ./ denom_RL;
    RL = RL .* real(ifft2( fft2(ratio) .* conj(H) ));

    RL = max(RL, 0);  % RL is usually constrained to be nonnegative
    psnrRL(i) = psnr(RL, f);

    %% ---------- Landweber ----------
    % Lw_{n+1} = Lw_n + h(-x) ⊗ ( g - (Lw_n ⊗ h) )
    residual = g - blur(Lw);
    Lw = Lw + real(ifft2( fft2(residual) .* conj(H) ));

    Lw = min(max(Lw, 0), 1);
    psnrLw(i) = psnr(Lw, f);

    %% ---------- ISRA ----------
    % I_{n+1} = I_n .* [ (h(-x) ⊗ g) ./ (h(-x) ⊗ (h ⊗ I_n)) ]
    %
    % Numerator: h(-x) ⊗ g
    num = real(ifft2( fft2(g) .* conj(H) ));

    % Denominator: h(-x) ⊗ (h ⊗ I_n)
    tmp = blur(ISRA);   % h ⊗ I_n
    den = real(ifft2( fft2(tmp) .* conj(H) ));

    den = max(den, 1e-12);  % avoid division by zero

    ISRA = ISRA .* (num ./ den);

    ISRA = max(ISRA, 0);
    psnrIS(i) = psnr(ISRA, f);

    %% print occasionally
    if mod(i, 20) == 0
        fprintf("iter=%d  PSNR: RL=%.2f  Lw=%.2f  ISRA=%.2f\n", ...
            i, psnrRL(i), psnrLw(i), psnrIS(i));
    end
end

%%  11) Show final restored images
figure("Name", "Task 4b: final reconstructions", "NumberTitle","off");
imshow([f, g, WienerRec, Lw, RL, ISRA], []);
title("f (GT) | g (blur+noise) | Wiener | Landweber | Richardson-Lucy | ISRA");

%% 12) PSNR curves (rubric requirement)

figure("Name", "Task 4b: PSNR vs Iteration", "NumberTitle","off");
plot(psnrW,  "LineWidth", 1.5); hold on;
plot(psnrLw, "LineWidth", 1.5);
plot(psnrRL, "LineWidth", 1.5);
plot(psnrIS, "LineWidth", 1.5);

xlabel("Iteration");
ylabel("PSNR (dB)");
title(sprintf("PSNR comparison (%s)", caseName));
legend("Wiener", "Landweber", "Richardson-Lucy", "ISRA");
grid on;

%% 13) Print quantitative summary
fprintf("\n================ Task 4b Summary ================\n");
fprintf("Case: %s\n", caseName);
fprintf("Wiener final PSNR: %.2f dB\n", psnr(WienerRec, f));
fprintf("Landweber best PSNR: %.2f dB\n", max(psnrLw));
fprintf("RL best PSNR: %.2f dB\n", max(psnrRL));
fprintf("ISRA best PSNR: %.2f dB\n", max(psnrIS));
fprintf("=================================================\n");
