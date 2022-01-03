function [obj_val, target] = OT_grad_descent(max_grad_descent_steps, lambda,...
    C, K, epsilon, source, target)

    for opt_step = 1:max_grad_descent_steps 
        [target, grad_norm, obj_val] = OT_grad_descent_step(lambda, C, K, epsilon, source, target);
        if grad_norm < 10^-7
            break
        end
    end
