function [obj_val, target] = LP_grad_descent(max_grad_descent_steps, lambda,...
    source, target, p)

    for opt_step = 1:max_grad_descent_steps 
        [target, grad_norm, obj_val] = LP_grad_descent_step(lambda, source, target, p);
        if grad_norm < 10^-7
            break
        end
    end

    if max_grad_descent_steps == 0
        obj_val = norm(source-target,p) + lambda*entropy1D(target);
    end