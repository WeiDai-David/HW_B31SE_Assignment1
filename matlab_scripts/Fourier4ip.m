% Load the image
original_image = imread('../images/eye-hand.png');

% Convert to grayscale
gray_image = im2gray(original_image);

% Apply Fourier transform
fft_image = fft2(double(gray_image));

% Create a filter (e.g., high-pass filter)
filter = ones(size(fft_image));
filter(1:10, 1:10) = 0; % Example: block low frequencies

% Apply filter
filtered_fft_image = fft_image .* filter;

% Inverse Fourier transform
filtered_image = ifft2(filtered_fft_image);

% Display the original and filtered images
figure;
subplot(1, 2, 1);
imshow(gray_image);
title('Original Image');

subplot(1, 2, 2);
imshow(abs(filtered_image), []);
title('Filtered Image');