function target_2d = get_rand_peak(image_width, image_height)
    [X, Y] = meshgrid(linspace(-1,1,image_width), linspace(-1,1,image_height));
    sumXY = abs(X)+abs(Y);
    target_2d = max(1.- rand() * sumXY, 0.);
    x_shift = randi([0,image_width]);
    y_shift = randi([0,image_height]);
    target_2d = circshift(target_2d, x_shift, 1);
    target_2d = circshift(target_2d, y_shift, 2);
    if x_shift<image_width/2.
        target_2d(1:x_shift, :) = 0;
    else
        target_2d(x_shift:image_width, :) = 0;
    end
    
    if y_shift<image_height/2.
        target_2d(:, 1:y_shift) = 0;
    else
        target_2d(:, y_shift:image_height) = 0;
    end

    