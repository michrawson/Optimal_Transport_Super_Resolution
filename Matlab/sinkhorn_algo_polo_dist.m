function distW = sinkhorn_algo_polo_dist(C, K, epsilon, source, target)

    assert(all(isfinite(target)));

    n = size(C,1);
    u = ones(n,1);

    for opt_ind = 1:500
        
        u_new = source ./ (K*(target ./ (K'*u)));

        if not(all(isfinite(u_new)))
            break;
        end
        
        
        if max(abs(u - u_new)) < 10^(-7)
            u = u_new;
            break;
        else
            u = u_new;
        end
    end
        
    v = target./(K' * u);                
    T = spdiags(u,0,n,n) * K * spdiags(v,0,n,n);
    distW = norm(C * T, 'fro');
