function [distW, v] = sinkhorn_algo(C, K, source, target)

    n = size(C,1);
    v = ones(n,1);

    distW = 99999999;
    u = -1;
    
    for opt_ind = 1:5000
        u_new = source./(K * v);
        if sum(abs(u_new - u))< 10^(-7)
            break;
        end
        
        u = u_new;
        v = target./(K' * u);                
        T = spdiags(u,0,n,n) * K * spdiags(v,0,n,n);

        if not(all(isfinite(u)) && all(isfinite(v)))
            break;
        end

        distW = norm(C .* T, 'fro');
        assert(false);
    end
