clear
close all

load('star_data.mat', 'predictions', 'images');
nnet_predictions = predictions'; % 1 is cluster, 2 is noncluster

sample_inds = randsample(size(images,1), 16, false);

nnet_predictions_sample = nnet_predictions(sample_inds,1);
images_sample = images(sample_inds,:,:,:);

lambda_v = 10.^(-4:-2); % .001; 
lambda_len = length(lambda_v);

OT_epsilon_v = [.1, 1, 10];
OT_epsilon_len = length(OT_epsilon_v);
OT_prediction_class = zeros(size(images_sample,1), lambda_len, ...
                                                OT_epsilon_len);

tic
parfor data_ind = 1:size(images_sample,1)
%     data_ind
%     nnet_predictions(data_ind)
    for lambda_ind = 1:lambda_len
        lambda = lambda_v(lambda_ind);
        for OT_epsilon_ind = 1:OT_epsilon_len
            p = zeros(5,1);

            for channel = 1:5
                star_image = squeeze(images_sample(data_ind,channel,:,:));
                p(channel) = OT_start_prediction(star_image, lambda, ...
                                            OT_epsilon_v(OT_epsilon_ind));
            end
            
%         p = mean(p);
            p = max(p);

            if p==1
                OT_prediction_class(data_ind, lambda_ind, OT_epsilon_ind) = 1;
            else
                OT_prediction_class(data_ind, lambda_ind, OT_epsilon_ind) = 2;
            end
        end
    end
end

vecnorm(nnet_predictions_sample-OT_prediction_class,1,1)

toc
