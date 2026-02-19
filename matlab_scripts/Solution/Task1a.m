close all; clear; clc;

%% Task 1a: Nonlinear image filtering (iterative)
% I0(x,y) = I(x,y)
% In+1(x,y) = sum_{i=-1..1, j=-1..1} w_ij * In(x+i, y+j) / sum w_ij
% w_ij = exp( -k * | In(x,y) - In(x+i, y+j) | )
%
% Notes:
% - weights depend on pixel position (x,y) and iteration n

%% Choose test image
% Make sure they are grayscale. If RGB, convert to gray.
% imgPath = '../../images/trui.tif';
% imgPath = '../../images/baboon.tif';
% imgPath = '../../images/barbara.tif';
% imgPath = '../../images/cameraman.tif';
% imgPath = '../../images/emu.tif';
% imgPath = '../../images/newborn.tif';
imgPath = '../../images/pout.tif';
% baboon.tif、barbara.tif、cameraman.tif、emu.tif、newborn.tif、pout.tif

I = im2double(imread(imgPath));
if ndims(I) == 3
    I = rgb2gray(I);
end

[rows, cols] = size(I);

%% Experiment settings
% At least 3-4 variations required by rubric.
k_list    = [1, 10, 50, 100, 500];     % try different k values
N_list    = [0, 5, 10, 20, 50, 100];      % try different iteration numbers

% For fair comparison, we compute:
% 1) simple averaging iterative smoothing
% 2) nonlinear filtering iterative smoothing

%% Part A: Simple averaging (baseline)
% This is the simplest case when all weights are equal: w_ij = 1.

fprintf("Running baseline: simple averaging (multiple N)...\n");

% We also try multiple iteration numbers (same idea as nonlinear experiments)
N_list_avg = [0, 1, 5, 10, 20, 50, 100];

% Store baseline results for different iteration counts
results_avg = cell(1, length(N_list_avg));

for nn = 1:length(N_list_avg)

    N_baseline = N_list_avg(nn);
    fprintf("  Averaging: N = %d iterations\n", N_baseline);

    im1_avg = I;
    im2_avg = zeros(rows, cols);

    for iter = 1 : N_baseline

        for x = 2 : rows-1
            for y = 2 : cols-1

                s = 0;

                % 3x3 neighborhood sum
                for i = -1 : 1
                    for j = -1 : 1
                        s = s + im1_avg(x+i, y+j);
                    end
                end

                % divide by 9 -> mean
                im2_avg(x,y) = s / 9;
            end
        end

        % update for next iteration
        im1_avg = im2_avg;
        im2_avg = zeros(rows, cols);

    end

    % store result for this N
    results_avg{nn} = im1_avg;

end


%% Part B: Nonlinear iterative filtering (main task)
% Key difference:
% - Instead of equal weights, we compute:
%   w_ij = exp( -k * |center - neighbor| )
% - Then do normalized weighted average.

% will store results for multiple (k, N) experiments.
results = cell(length(k_list), length(N_list));

fprintf("Running nonlinear filtering experiments...\n");

for kk = 1:length(k_list)
    k = k_list(kk);

    for nn = 1:length(N_list)
        N = N_list(nn);

        fprintf("  k = %d, N = %d iterations\n", k, N);

        % Initialize iteration
        im1 = I;                     % In
        im2 = zeros(rows, cols);     % In+1

        for iter = 1:N

            % For each inner pixel
            for x = 2:rows-1
                for y = 2:cols-1

                    center = im1(x,y);

                    weighted_sum = 0;
                    weight_total = 0;

                    % 3x3 neighborhood
                    for i = -1:1
                        for j = -1:1

                            neighbor = im1(x+i, y+j);

                            % Nonlinear weight (depends on current iter image)
                            w = exp( -k * abs(center - neighbor) );

                            weighted_sum = weighted_sum + w * neighbor;
                            weight_total = weight_total + w;

                        end
                    end

                    % Normalized weighted average
                    im2(x,y) = weighted_sum / weight_total;

                end
            end

            % Update for next iteration
            im1 = im2;
            im2 = zeros(rows, cols);
        end

        results{kk, nn} = im1;
    end
end

%% Visualization
% We show:
% 1) original image
% 2) baseline averaging
% 3) nonlinear results for multiple k and N

% --- Show baseline averaging results in a grid ---
figure('Name', 'Simple averaging results (multiple N)');

tiledlayout(1, length(N_list_avg), 'Padding', 'compact', 'TileSpacing', 'compact');

for nn = 1:length(N_list_avg)
    nexttile;
    imshow(results_avg{nn}, []);
    title(sprintf("Avg N=%d", N_list_avg(nn)));
end

% --- Show nonlinear results in a grid ---
figure('Name', 'Nonlinear filtering results');

tiledlayout(length(k_list), length(N_list), 'Padding', 'compact', 'TileSpacing', 'compact');

for kk = 1:length(k_list)
    for nn = 1:length(N_list)

        nexttile;
        imshow(results{kk, nn}, []);

        title(sprintf("k=%d, N=%d", k_list(kk), N_list(nn)));
    end
end
