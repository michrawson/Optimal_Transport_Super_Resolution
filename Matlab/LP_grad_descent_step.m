function [target, grad_norm, obj_val] = LP_grad_descent_step(lambda, source, ...
    target, p)

    dNorm_target = zeros(length(target), 1);

    if p == 1
        for i = 1:length(target)
            if source(i,1)-target(i,1) >= 0
                dNorm_target(i,1) = -1;
            elseif source(i,1)-target(i,1) < 0
                dNorm_target(i,1) = 1;
            end
        end
    elseif p == 2
        for i = 1:length(target)
            dNorm_target(i,1) = 2*(source(i,1)-target(i,1))*-1;
        end
    else
        assert(false);
    end
    
%     dNorm_target = dNorm_target ...
%         - dot(dNorm_target,ones(length(target),1))*dNorm_target;
    
    dEntropy = D_entropy1D(target);
    
%     dEntropy = dEntropy - dot(dEntropy,ones(length(target),1))*dEntropy;
    
    grad = dNorm_target + lambda*dEntropy;
    
    grad = grad - dot(grad,ones(length(target),1))*grad;
    
    grad_norm = norm(grad,1);
    
    obj_val = norm(source-target,p) + lambda*entropy1D(target);
    
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

            obj_val = norm(source-target_new,p) + lambda*entropy1D(target_new);

            if not(isfinite(obj_val))
                obj_val = obj_val_prev;
            end
            step_size = step_size/2;
        end
        if obj_val < obj_val_prev 
            target = target_new;
        end
    end
    