function [distW, dW] = sinkhorn_algo_polo(C, K, epsilon, source, target, opt_iters)

    assert(all(isfinite(target)));

    n = size(C,1);
    u = ones(n,1);

    for opt_ind = 1:opt_iters
        
        u_new = source ./ (K*(target ./ (K'*u)));

        if not(all(isfinite(u_new)))
            distW = 9999999;
            dW = zeros(n,1);
            return
        end
        
        
        if max(abs(u - u_new)) < 10^(-7)
            u = u_new;
            break;
        else
            u = u_new;
        end
    end
    
    u_pos = u;
    u_pos(u_pos<10^-7) = 10^-7;
    
    u_ones = u'*ones(n,1);
    u_ones(u_ones<10^-7) = 10^-7;
        
    l_u_ones_k = log(u_ones)./K;
    l_u_ones_k(l_u_ones_k>9999999)=9999999;
    assert(all(isfinite(l_u_ones_k),'all'));
    
    dW = epsilon*log(u_pos) + epsilon*l_u_ones_k*ones(n,1);
    v = target./(K' * u);                
    T = spdiags(u,0,n,n) * K * spdiags(v,0,n,n);
    distW = sqrt(sum((C.^2) .* T,'all'));
    
    assert(all(isfinite(dW)));
    