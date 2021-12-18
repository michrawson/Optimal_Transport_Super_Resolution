function target_W_points = OT_start_prediction(star_image, lambda, OT_epsilon)

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
%     C = double(C);
    
    K = exp(-C./OT_epsilon);
%     K = double(K);
    
    targets_size = 100;
    
    dist_W = zeros(targets_size,1);
    target_v = zeros(targets_size,n);

    for rand_ind = 1:targets_size

        if rand() > .5
            target_2d = get_rand_peak(image_width, image_height);
        else
            target_2d = get_rand_peak(image_width, image_height) ...
                        + get_rand_peak(image_width, image_height);
        end
        target = reshape(target_2d, image_width*image_height, 1);
        target = (target/norm(target,1));
        
        distW = sinkhorn_algo(C, K, source, target);

        dist_W(rand_ind,1) = distW ...
            + lambda*entropy1D(target);
        target_v(rand_ind,:) = target;
        
    end
    [M,I] = min(dist_W);
    optimal_target = target_v(I,:);
    target_W_points = L0_2D(reshape(optimal_target, image_width, image_height));
  
