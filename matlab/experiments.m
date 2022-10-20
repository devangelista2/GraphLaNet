
%close all
clearvars -except A

load('A')
load('280.mat')


gt=double(gt);
recon=double(recon);

y=A*sparse(gt(:));
y=full(reshape(y,367,120));
rng(7);
y_noise = y + 0.5*randn(size(y));


mu=500;
delta=0.001;
R=3;

optionsTV=struct('mu',mu,'q',1,'n',256,'m',256,'sigmaInt',delta,'R',R);
xTV=l2lqFract(A,y_noise,optionsTV);
% 
% options=struct('recon',recon,'mu',mu,'q',1,'n',256,'m',256,...
%     'sigmaInt',delta,'R',R);
% xLNN=l2lqFractNN(A,y_noise,options);
% 
% optionsgt=struct('recon',gt,'mu',mu,'q',1,'n',256,'m',256,...
%     'sigmaInt',delta,'R',R);
% xgt=l2lqFractNN(A,y_noise,optionsgt);
% 
% 
% 
% 
% 
% imshow(gt)
% figure
% imshow(xTV/max(xTV(:)))
% figure
% imshow(xLNN/max(xLNN(:)))
% figure
% imshow(recon/max(recon(:)))
% 
% eNN=norm(recon-gt,'fro')/norm(gt,'fro')
% eTV=norm(xTV-gt,'fro')/norm(gt,'fro')
% eLNN=norm(xLNN-gt,'fro')/norm(gt,'fro')
% egt=norm(xgt-gt,'fro')/norm(gt,'fro')
% 
% 
% peaksnrNN = psnr(recon,gt)
% peaksnrTV = psnr(xTV,gt)
% peaksnrLNN = psnr(xLNN,gt)
% peaksnrgt = psnr(xgt,gt)
% 
% ssimvalNN = ssim(recon,gt)
% ssimvalTV = ssim(xTV,gt)
% ssimvalLNN = ssim(xLNN,gt)
% ssimvalgt = ssim(xgt,gt)



