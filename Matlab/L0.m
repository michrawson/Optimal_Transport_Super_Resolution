function y = L0(x, thresh)

x_abs = abs(x);

thresh = max(x_abs)*thresh;

x_large_ind = x_abs>thresh;

y = 0;
for i = 1:length(x_large_ind)
    if i == 1
        if x_large_ind(i)==1
            y = y + 1;
        end
    else
        if x_large_ind(i)==1 && x_large_ind(i-1)==0
            y = y + 1;
        end
    end
end

