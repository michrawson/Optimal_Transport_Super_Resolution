function target_W_points = OT_start_prediction(star_image, lambda_v, OT_epsilon)

    image_width = size(star_image, 1);
    image_height = size(star_image, 2);
    source = reshape(star_image, image_width*image_height, 1);
    source = source/norm(source,1);

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

    targets_size = 100;
    
    dist_W = zeros(targets_size,length(lambda_v));
    dist_l1 = zeros(size(dist_W));
    dist_l2 = zeros(size(dist_W));
    target_v = zeros(targets_size,length(lambda_v),n);
    target_W_entropy = zeros(length(lambda_v),1);
    target_W_points = zeros(length(lambda_v),1);

    for lambda_ind = 1:length(lambda_v)
        for rand_ind = 1:targets_size
            
            if rand() > .5
                target_2d = get_rand_peak(image_width, image_height);
            else
                target_2d = get_rand_peak(image_width, image_height) ...
                            + get_rand_peak(image_width, image_height);
            end
            target = reshape(target_2d, image_width*image_height, 1);
            target = target/norm(target,1);

            u = ones(n,1);
            v = ones(n,1);
            T = diag(u) * K * diag(v);

            distW = -1;
            for opt_ind = 1:1000
                u = source./(K * v);     
                v = target./(K' * u);                
                T = diag(u) * K * diag(v);
                
                if not(all(isfinite(u)) && all(isfinite(v)))
                    break;
                end
                
                if abs(norm(C * T, 'fro') - distW)<10^(-8)
                    break;
                end
                distW = norm(C * T, 'fro');
            end
            
            dist_W(rand_ind,lambda_ind) = norm(C * T, 'fro') ...
                + lambda_v(lambda_ind)*entropy1D(target);
            dist_l1(rand_ind,lambda_ind) = norm(source-target,1) ...
                + lambda_v(lambda_ind)*entropy1D(target);
            dist_l2(rand_ind,lambda_ind) = norm(source-target,2) ...
                + lambda_v(lambda_ind)*entropy1D(target);
            target_v(rand_ind,lambda_ind,:) = target;
        end
        [M,I] = min(dist_W(:,lambda_ind));
        optimal_target = target_v(I,lambda_ind,:);
        target_W_entropy(lambda_ind) = entropy1D(optimal_target);
        target_W_points(lambda_ind) = L0_2D(reshape(optimal_target, ...
                                                image_width, image_height));
    end
