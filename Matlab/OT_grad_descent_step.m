function [target, grad_norm, obj_val] = OT_grad_descent_step(lambda, C, K, ...
    epsilon, source, target, opt_iters)

    assert(all(isfinite(target)));    
    n = length(target);
    [distW, dW_target] = sinkhorn_algo_polo(C, K, epsilon, source, target, opt_iters);
        
    dEntropy = D_entropy1D(target);
            
    grad = dW_target + lambda*dEntropy;

    grad(not(isfinite(dEntropy)))=0;

    grad(isfinite(dEntropy)) = grad(isfinite(dEntropy)) ...
                                - sum(grad)/sum(isfinite(dEntropy));

    assert(abs(dot(grad,ones(n,1)/sqrt(n)))<10^-7)
    
    grad_norm = norm(grad,1);
    
    obj_val = distW + lambda*entropy1D(target);
    
    if not(isfinite(obj_val))
        obj_val = 999999999;
    end
    if all(isfinite(grad))

        obj_val_prev = obj_val;

        step_size = .01;
        while (obj_val >= obj_val_prev && step_size>10^-12)
            target_new = target - step_size*grad;

            if(all(target_new<=0))
                target_new = -target_new;
            end
            
            target_new(target_new<0)=0;
            
            assert(norm(target_new,1)>0);
            target_new = target_new/norm(target_new,1);
            
            distW = sinkhorn_algo_polo_dist(C, K, epsilon, source, target_new, opt_iters);
            obj_val = distW + lambda*entropy1D(target_new);
            if not(isfinite(obj_val))
                obj_val = obj_val_prev;
            end
            step_size = step_size/2;
        end
        if obj_val < obj_val_prev 
            target = target_new;
        end
    end
    
    