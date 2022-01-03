function e = D_entropy1D(X)

    e = zeros(length(X),1);
    for i = 1:length(X)
        if X(i)>0
            e(i,1) = log(X(i)) + 1;
        else
            e(i,1) = -9999999;
        end
    end
    assert(all(isfinite(e)));
    