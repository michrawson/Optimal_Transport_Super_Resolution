close all
clear

n = 5;

line_spec = {'o','+','*','x','|','s','d','^','v','>','<','p','h'};

C = zeros(n,n);
for i = 1:n
    for j = 1:n
        C(i,j) = abs(i-j);
    end
end

source = (randn(n,1)).^2;
source = source/sum(source);

target = (randn(n,1)).^2;
target = target/sum(target);

norm(source-target,1)
norm(source-target,2)
norm(source-target,inf)

epsilon_v = 10.^(-4:.5:0);
for ep_iter = 1:length(epsilon_v)
    epsilon = epsilon_v(ep_iter);

    K = exp(-C/epsilon);

    u = ones(n,1);
    v = ones(n,1);
    T = diag(u) * K * diag(v);

    iterations = 160;
    distW = zeros(iterations,1);
    for opt_ind = 1:iterations
        u = source./(K*v);
        v = target./(K'*u);
        T = diag(u) * K * diag(v);
        distW(opt_ind) = norm(C .* T, 'fro');
    end
    x_range = 1:3:iterations;
    plot(x_range, distW(x_range), strcat('-',line_spec{ep_iter}), 'LineWidth',1);
    legend_names{ep_iter} = sprintf('\\epsilon=%2.4f',epsilon);
    hold on;
end
legend(legend_names);
ylabel('distance');
xlabel('iterations');

