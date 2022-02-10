function [obj_val, target] = OT_grad_descent(max_grad_descent_steps, lambda,...
    C, K, epsilon, source, target, opt_iters, init_step_size)

    assert(all(isfinite(target)));
    for opt_step = 1:max_grad_descent_steps 
        [target, grad_norm, obj_val] = OT_grad_descent_step(lambda, C, K, ...
            epsilon, source, target, opt_iters, init_step_size);
        assert(all(isfinite(target)));
%         fprintf('%d, %2.4f\n',opt_step,grad_norm)
        if grad_norm < 10^-7
            break
        end
    end

    if max_grad_descent_steps == 0
        distW = sinkhorn_algo_polo_dist(C, K, epsilon, source, target, opt_iters);
        obj_val = distW + lambda*entropy1D(target);
    end
    