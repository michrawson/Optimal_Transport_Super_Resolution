close all
clear

n = 5;
epsilon = .01;
sig_to_noise = 0; 
thresh = 0.22;

C = zeros(n,n);
for i = 1:n
    for j = 1:n
        C(i,j) = abs(i-j);
    end
end

K = exp(-C/epsilon);

source_clean = zeros(n,1);
source_clean(1) = 1;
source_clean(n) = 1/2.;
source_clean = source_clean/sum(source_clean);

source = source_clean + sig_to_noise*rand(n,1);
source(3) = 1/5.;
source(source<0)=0;
source = source/sum(source);

figure
plot(1:n, source_clean, ...
     1:n, source, 'LineWidth',1)
legend('signal','signal+noise')

mesh_size = 5;
mesh_v = linspace(0,1,mesh_size);
targets_size = length(mesh_v)^n;

lambda_v = .6:.1:1.5;

dist_W = zeros(targets_size,length(lambda_v));
dist_l1 = zeros(targets_size,length(lambda_v));
dist_l2 = zeros(targets_size,length(lambda_v));
target_v = zeros(targets_size,length(lambda_v),n);
target_W_dist = zeros(length(lambda_v),1);
target_W_entropy = zeros(length(lambda_v),1);
target_L1_entropy = zeros(length(lambda_v),1);
target_L2_entropy = zeros(length(lambda_v),1);
target_W_points = zeros(length(lambda_v),1);
target_L1_points = zeros(length(lambda_v),1);
target_L2_points = zeros(length(lambda_v),1);
for lambda_ind = 1:length(lambda_v)
    
    targets = deterministic_dist(n,mesh_v);
    parfor rand_ind = 1:targets_size % stat_sig
        target = targets(:,rand_ind);
        
        u = ones(n,1);
        v = ones(n,1);
        T = diag(u) * K * diag(v);

        for opt_ind = 1:1000
            u = source./(K*v);
            v = target./(K'*u);
            T = diag(u) * K * diag(v);
        end
        
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
    target_W_dist(lambda_ind) = dist_W(I,lambda_ind);
    
    [M,I2] = min(dist_l1(:,lambda_ind));
    optimal_target = target_v(I2,lambda_ind,:);
    target_L1_entropy(lambda_ind) = entropy1D(optimal_target);
    target_L1_points(lambda_ind) = L0(optimal_target, thresh);
    
    [M,I3] = min(dist_l2(:,lambda_ind));
    optimal_target = target_v(I3,lambda_ind,:);
    target_L2_entropy(lambda_ind) = entropy1D(optimal_target);
    target_L2_points(lambda_ind) = L0(optimal_target, thresh);

end

figure
yyaxis left
plot(lambda_v,target_W_points,'-o','LineWidth',1);
xlabel('\lambda')
ylabel('|v|_M')

yyaxis right
plot(lambda_v, target_W_dist,'-d','LineWidth',1)
ylabel('d_W(\mu+\eta,v)')

