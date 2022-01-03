function [distW, v] = sinkhorn_algo(C, K, source, target)

    n = size(C,1);
    v = ones(n,1);

    distW = -1;
    for opt_ind = 1:500
        u = source./(K * v);     
        v = target./(K' * u);                
        T = spdiags(u,0,n,n) * K * spdiags(v,0,n,n);

        if not(all(isfinite(u)) && all(isfinite(v)))
            break;
        end

        distWnew = norm(C * T, 'fro');
        if abs(distWnew - distW)< 10^(-7)
            distW = distWnew;
            break;
        else
            distW = distWnew;
        end
    end
