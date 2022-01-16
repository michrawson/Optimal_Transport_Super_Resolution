clear
close all

load('star_data.mat', 'predictions', 'images');
nnet_predictions = predictions'; % 1 is cluster, 2 is noncluster

images = images - min(images,[],'all');
assert(all(images>=0,'all'));

num_channels = 5;
max_grad_descent_steps = 4;
targets_size = 1;%100;
opt_iters = 5000;
for L0_thresh = [.5, .9]
lambda_v = 10.^(-3:0); 
OT_epsilon_v = 10.^(-3:0);

% sample_inds = randsample(size(images,1), 10, false);
sample_inds = 1:8;

% for i = sample_inds
%     for j = 1:num_channels
%         figure
%         imagesc(squeeze(images(i,j,:,:)))
%     end
% end
% 
% return

nnet_predictions_sample = nnet_predictions(sample_inds,1);

images_sample = images(sample_inds,:,:,:);

lambda_len = length(lambda_v);

OT_epsilon_len = length(OT_epsilon_v);

experiments = {};
c=1;
for data_ind = 1:size(images_sample,1)
    for lambda_ind = 1:lambda_len
        lambda = lambda_v(lambda_ind);
        for OT_epsilon_ind = 1:OT_epsilon_len
            OT_epsilon = OT_epsilon_v(OT_epsilon_ind);
            for channel = 1:num_channels
                experiments{c} = [data_ind, lambda_ind, OT_epsilon_ind, channel];
                c = c + 1;
            end
        end
    end
end

experiments_results = {};

tic
parfor exper_ind = 1:length(experiments) % parfor
    e = experiments{exper_ind};
    data_ind = e(1);
    lambda_ind = e(2);
    OT_epsilon_ind = e(3);
    channel = e(4);

    star_image = squeeze(images_sample(data_ind,channel,:,:));
    lambda = lambda_v(lambda_ind);
    OT_epsilon = OT_epsilon_v(OT_epsilon_ind);
    
    experiments_results{exper_ind} = OT_start_prediction(star_image, lambda, ...
                OT_epsilon, max_grad_descent_steps, targets_size, L0_thresh, opt_iters);

end
toc

OT_prediction_class_channels = zeros(size(images_sample,1), lambda_len, ...
                                                OT_epsilon_len, num_channels);

for exper_ind = 1:length(experiments)
    e = experiments{exper_ind};
    data_ind = e(1);
    lambda_ind = e(2);
    OT_epsilon_ind = e(3);
    channel = e(4);
    
    OT_prediction_class_channels(data_ind, lambda_ind, OT_epsilon_ind, channel) ...
        = experiments_results{exper_ind};
end

OT_prediction_class_channels_class = OT_prediction_class_channels;
OT_prediction_class_channels_class(OT_prediction_class_channels_class>1)=2;

% OT_prediction_class = min(OT_prediction_class_channels_class,[],4);
OT_prediction_class = round(mean(OT_prediction_class_channels_class,4));

pred_diff = nnet_predictions_sample-OT_prediction_class;

sum(abs(pred_diff),1)

% vecnorm(pred_diff,1,1)
% 
% pred_diff_pos = pred_diff;
% pred_diff_pos(pred_diff_pos<0)=0;
% 
% vecnorm(pred_diff_pos,1,1)
% 
% pred_diff_neg = pred_diff;
% pred_diff_neg(pred_diff_neg>0)=0;
% 
% vecnorm(pred_diff_neg,1,1)
end

