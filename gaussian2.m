function [ G ] = gaussian2(w, sd )
%GAUSSIAN2 returns a 2D Gaussian mask
%   Gaussian mask is centred on an W x W matrix, with standard deviation SD

w = ceil(w);
w = 2*floor(w/2)+1;         % force odd size mask
        c = (w+1)/2;        % central point of mask
        x = (1/(sd*sqrt(2*pi)))*exp(-((1:w)-c).^2/(2*sd*sd));
        x = x/sum(x);                  % normalise
        G = x'*x;
end

