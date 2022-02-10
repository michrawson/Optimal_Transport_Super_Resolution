function distW = sinkhorn_algo_polo_dist(C, K, epsilon, source, target, opt_iters)

    assert(all(isfinite(target)));

    if max(abs(source-target))<10^-7
        distW=0;
        return
    end
    
    n = size(C,1);
    u = ones(n,1);

    for opt_ind = 1:opt_iters
        
        u_new = source ./ (K*(target ./ (K'*u)));

        if not(all(isfinite(u_new)))
            distW = 9999999;
            return
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
%     distW = sqrt(sum((C.^2) .* T,'all'));
    distW = sum(C .* T,'all');