function e = entropy1D(X)

    e = 0;
    for i = 1:length(X)
        if X(i)>0
            e = e - (X(i) * log(X(i)));
        end
    end
    assert(isfinite(e));
    