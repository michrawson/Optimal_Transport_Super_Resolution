% close all
clear

n = 5;
epsilon = .1;
sig_to_noise = .3; % 1 = high noise, 0 = no noise

C = zeros(n,n);
for i = 1:n
    for j = 1:n
        C(i,j) = abs(i-j)/4;
    end
end

% C = ones(n,n) - eye(n);

K = exp(-C/epsilon);
% K = eye(n);

% source = [rand(n-1,1)/(n-1); 0];
% source(end) = 1-sum(source);

source = zeros(n,1);
% source(randperm(n,2))=1;
source = source + sig_to_noise*rand(n,1);
source(1) = 1;

% source(3) = 1/2;

source(n) = 1;
source = source/sum(source);

% figure
% plot(source)

mesh_size = 5;
mesh_v = linspace(0,1,mesh_size);
targets_size = length(mesh_v)^n;
% stat_sig = 10000;

% lambda_v = 10.^(-12:1:10);
lambda_v = 10.^(-2:.1:2);
for thresh = .1
thresh

dist_W = zeros(targets_size,length(lambda_v));
dist_l1 = zeros(targets_size,length(lambda_v));
dist_l2 = zeros(targets_size,length(lambda_v));
target_v = zeros(targets_size,length(lambda_v),n);
target_W_entropy = zeros(length(lambda_v),1);
target_L1_entropy = zeros(length(lambda_v),1);
target_L2_entropy = zeros(length(lambda_v),1);
target_W_points = zeros(length(lambda_v),1);
target_L1_points = zeros(length(lambda_v),1);
target_L2_points = zeros(length(lambda_v),1);
for lambda_ind = 1:length(lambda_v)
    
    targets = deterministic_dist(n,mesh_v);
    parfor rand_ind = 1:targets_size % stat_sig
%         target = rand_dist(n);
        target = targets(:,rand_ind);
        
        u = ones(n,1);
        v = ones(n,1);
        T = diag(u) * K * diag(v);

        for opt_ind = 1:1000
            u = source./(K*v);
            v = target./(K'*u);
            T = diag(u) * K * diag(v);
        end
%         norm(C .* T, 'fro')
        
        dist_W(rand_ind,lambda_ind) = norm(C .* T, 'fro') ...
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
    target_W_points(lambda_ind) = L0(optimal_target, thresh);

%     figure;
%     plot_sq(optimal_target);
%     title('W');
    
    [M,I2] = min(dist_l1(:,lambda_ind));
    optimal_target = target_v(I2,lambda_ind,:);
    target_L1_entropy(lambda_ind) = entropy1D(optimal_target);
    target_L1_points(lambda_ind) = L0(optimal_target, thresh);

%     figure;
%     plot_sq(optimal_target);
%     title('L1');
    
    [M,I3] = min(dist_l2(:,lambda_ind));
    optimal_target = target_v(I3,lambda_ind,:);
    target_L2_entropy(lambda_ind) = entropy1D(optimal_target);
    target_L2_points(lambda_ind) = L0(optimal_target, thresh);

%     figure;
%     plot_sq(optimal_target);
%     title('L2');
%     return
end

figure
semilogx(   lambda_v,target_W_entropy,'-*',...
            lambda_v,target_L1_entropy,'-o',...
            lambda_v,target_L2_entropy,'-s')
xlabel('\lambda')
ylabel('entropy(v)')
legend('d=W1','d=L1','d=L2')

figure
semilogx(   lambda_v,exp(target_W_entropy),'-*',...
            lambda_v,exp(target_L1_entropy),'-o',...
            lambda_v,exp(target_L2_entropy),'-s')
xlabel('\lambda')
ylabel('exp(entropy(v))')
legend('d=W1','d=L1','d=L2')

figure
semilogx(   lambda_v,target_W_points,'-*',...
            lambda_v,target_L1_points,'-o',...
            lambda_v,target_L2_points,'-s')
xlabel('\lambda')
ylabel('L0(v)')
legend('d=W1','d=L1','d=L2')
end