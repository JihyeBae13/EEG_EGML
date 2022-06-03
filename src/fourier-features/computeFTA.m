function X_fta = computeFTA(x, n)
if ~exist('n', 'var')
    n = ceil(size(x, 1) / 2);
end
X_fta = zeros(size(x));
X_fta = X_fta(1:(2*n-1), :, :);
% if mod(size(x, 1) , 2) == 0
%    X_fta(end, :,:) = [];
% end
x_f = fft(x);
X_fta(1:2:(2*n-1), :, :) = real(x_f(1:n,:,:)); 
X_fta(2:2:(2*n-1), :, :) = imag(x_f(2:n, :, :));

end
