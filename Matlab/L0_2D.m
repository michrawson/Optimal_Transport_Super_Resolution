function y = L0_2D(x, L0_thresh)
    x_abs = abs(x);
    thresh = max(x_abs,[],'all')*L0_thresh;
    y = 0;
    for i = 1:size(x_abs,1)
        for j = 1:size(x_abs,2)
            if x_abs(i,j)>thresh
                if i == 1 && j == 1
                    y = y + 1;
                elseif i == 1
                    if x_abs(i,j-1)<=thresh
                        y = y + 1;
                    end
                elseif j == 1
                    if x_abs(i-1,j)<=thresh && x_abs(i-1,j+1)<=thresh
                        y = y + 1;
                    end
                elseif j == size(x_abs,2)
                    if x_abs(i-1,j)<=thresh && x_abs(i,j-1)<=thresh ...
                                        && x_abs(i-1,j-1)<=thresh
                        y = y + 1;                   
                    end
                else
                    if x_abs(i-1,j)<=thresh && x_abs(i,j-1)<=thresh ...
                            && x_abs(i-1,j-1)<=thresh ...
                            && x_abs(i-1,j+1)<=thresh
                        y = y + 1;
                    end
                end
            end
        end
    end

    