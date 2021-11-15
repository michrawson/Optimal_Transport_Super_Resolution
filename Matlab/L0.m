function y = L0(x, thresh)

x_abs = abs(x);

% thresh = max(x_abs)*0.5;

x_large = x_abs>thresh;

y = 0;
for i = 1:length(x_large)
    if i == 1
        if x_large(i)==1
            y = y + 1;
        end
    else
        if x_large(i)==1 && x_large(i-1)==0
            y = y + 1;
        end
    end
end

% y = sum(x_abs>thresh);

% y = sum(abs(sign(x)));
