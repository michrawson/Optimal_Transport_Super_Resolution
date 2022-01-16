function target_W_points = OT_start_prediction(star_image, lambda, OT_epsilon, ...
    max_grad_descent_steps, targets_size, L0_thresh)

    assert(all(star_image>=0,'all'));
    assert(sum(abs(star_image),'all')>0);

    image_width = size(star_image, 1);
    image_height = size(star_image, 2);
    source = reshape(star_image, image_width*image_height, 1);
    source = (source/norm(source,1));

    n = length(source);

    C = zeros(n,n);
    for i = 1:n
        for j = 1:n
            [ix, iy] = ind2sub([image_width, image_height], i);
            [jx, jy] = ind2sub([image_width, image_height], j);
            C(i,j) = norm([ix-jx, iy-jy]);
        end
    end
    
    K = exp(-C./OT_epsilon);
    if(not(all(K>0, 'all')))
        target_W_points = nan;
        return
    end
    
    obj_val_v = zeros(targets_size,1);
    target_v = zeros(targets_size,n);

    for rand_ind = 1:targets_size

%         if rand() > .5
%             target_2d = get_rand_peak(image_width, image_height);
%         else
%             target_2d = get_rand_peak(image_width, image_height) ...
%                         + get_rand_peak(image_width, image_height);
%         end
%         target = reshape(target_2d, image_width*image_height, 1);
%         target = (target/norm(target,1));

        target = source;% + .1*(2*rand(image_width*image_height,1)-1);
        
        target(target<0) = 0;
        
        target = target/norm(target,1);
        
        [obj_val, target_OT] = OT_grad_descent(max_grad_descent_steps, lambda,...
                                            C, K, OT_epsilon, source, target);
        
        obj_val_v(rand_ind,1) = obj_val;
        target_v(rand_ind,:) = target_OT;
        
%         distW = sinkhorn_algo(C, K, source, target);
% 
%         obj_val_v(rand_ind,1) = distW ...
%             + lambda*entropy1D(target);
%         target_v(rand_ind,:) = target;

    end
    [M,I] = min(obj_val_v);
    optimal_target = target_v(I,:);
    target_W_points = L0_2D(reshape(optimal_target, image_width, image_height),L0_thresh);


