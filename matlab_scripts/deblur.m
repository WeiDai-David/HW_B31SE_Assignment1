close all; clear; clc;

% h = double(imread('testkernel.png')); % motion blur
% h = fspecial('gaussian', [15 15], 5); % Gaussian blur
h = fspecial('gaussian', [9 9], 4); % Beck-Teboulle SIAM 2009
h = h./sum(h(:));

blur = @(im) imfilter(im,h,'conv','circular');


% Gaussian noise
noise_mean = 0;
% noise_var = 0.00001; % 10^{-5}
noise_var = 0.000001; % 10^{-6} Beck-Teboulle SIAM 2009 

% f = im2double(imread('barbara_face.png'));
f = im2double(imread('cameraman.tif'));
g = imfilter(f,h,'conv','circular'); % blur
g = imnoise(g,'gaussian',noise_mean,noise_var); % ading noise

H = psf2otf(h,size(g));

psnr0 = psnr(f,g);
psnrRL = [psnr0];
psnrLw = [psnr0];

% Wiener deblurring
W = deconvwnr(g,h,0.0001);
figure,imshow([f,g,W]);title('original, blurred and noisy, Wiener deblur');

RL = g;
Lw = g;
G = fft2(g);
maxiter = 3000; 
for i = 1:maxiter
 %
 % Richardson-Lucy iterations: RL = RL.*[h(-x)*(g./(RL*h(x)))]
 RL = RL.*ifft2(fft2(g./blur(RL)).*conj(H));
 psnr_RL = psnr(RL,f);
 psnrRL = [psnrRL; psnr_RL];
 %
 % Landweber iterations: Lw = Lw + h(-x)*(Lw-Lw*h(x))
 Lw = Lw + ifft2(conj(H).*(fft2(g-blur(Lw))));
 psnr_Lw = psnr(Lw,f);
 psnrLw = [psnrLw; psnr_Lw];
 %
 fprintf('i = %d   psnr_RL = %f   psnr_Lw = %f\n', i, psnr_RL, psnr_Lw);
 %
end

psnrW = psnr(W,f)*ones(maxiter,1);

figure,imshow([Lw,RL]);title('Landweber and Richardson-Lucy');

figure();
semilogy(psnrW,'LineWidth',1.5,'Color',[0,0,1]),axis([1 maxiter 0 30]); 
hold
semilogy(psnrLw,'LineWidth',1.5,'Color',[0,1,0]),axis([1 maxiter 0 35]);
semilogy(psnrRL,'LineWidth',1.5,'Color',[1,0,0]),axis([1 maxiter 0 30]);
legend('Wiener', 'Landweber','Richardson-Lucy');

