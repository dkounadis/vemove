% Example of using vemove.m for the separation of moving sound sources
clc
clear
close all

% contains auxilliary functions (STFT, KL_NMF, etc.) and is downloaded from
% http://www.irisa.fr/metiss/ozerov/Software/multi_nmf_toolbox.zip
addpath aux_tools/

mkdir results

J = 3;              % number of sources
cPerSrc = 20;       % components of NMF assigned to each source
stft_win_len = 512; % length of the STFT analysis window
maxNumIter = 20;    % number of EM iterations
snr = 10;           % level of corruption of parameters initialization (dB)
%snr = Inf;         % this is the perfect initialization (no corruption)

% {J} x [M x I] load J ground-truth source-images from ./data folder
[y,fs] = arrayfun(@(j) audioread(sprintf('trueSrc%d.wav',j)), 1:J, 'uniformoutput', false);

% [M x I x J] array of ground-truth source images
y = cat(3,y{:});

% [M x I] make the mixture signal by adding the true src-images
x = sum(y,3);

% M is the number of time samples, I is the number of mikes
[M,I] = size(x);

% [F x L x I] calculate the STFT of x
X = stft_multi( transpose(x), stft_win_len);

% write the mix in a .wav
audiowrite('./results/mix.wav', x/max(abs(x(:))) ,fs{1});

% initialization of NMF parameters (use one of the microphones)
[W,H,Kj] = corruptInit( y , cPerSrc, stft_win_len , snr);

fprintf('Applying separation ..\n\n');

% vemove() to do the separation
[A,S] = vemove(X,W,H,Kj,maxNumIter);

% [I x M x J] reconstruct the (estimated) time-domain src-img
ye = zeros(M,I,J);

for j=1:J
    ye(:,:,j) = transpose( istft_multi( bsxfun(@times,A(:,:,:,j),S(:,:,j)) , M ) );
end

% [M x I x J] normalise estimated src-img to avoid decliping when writing
ye = bsxfun(@rdivide,ye,max(abs(ye)));

arrayfun(@(j) audiowrite(sprintf('./results/estimatedSrc%d.wav',j),ye(:,:,j),fs{1}), 1:J , 'uniformoutput',false)

fprintf('\nSeparation results are written in ./results\n');


















