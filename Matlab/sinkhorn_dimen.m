close all
clear

line_spec = {'o','+','*','x','|','s','d','^','v','>','<','p','h'};

n_v = 3:10;
rand_v = 1:2000;
iters = zeros(length(rand_v),length(n_v));
for rand_ind = rand_v
    for n_ind = 1:length(n_v)
        n = n_v(n_ind);

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

        epsilon = .01;

        K = exp(-C/epsilon);

        u = ones(n,1);
        v = ones(n,1);
        T = diag(u) * K * diag(v);

        max_iterations = 10000;
        iterations = 1;
        distW = zeros(max_iterations,1);
        while iterations<3
            u = source./(K*v);
            v = target./(K'*u);
            T = diag(u) * K * diag(v);
            distW(iterations) = norm(C .* T, 'fro');
            iterations = iterations + 1;
        end
        while distW(iterations-2) ~= distW(iterations-1)
            u = source./(K*v);
            v = target./(K'*u);
            T = diag(u) * K * diag(v);
            distW(iterations) = norm(C .* T, 'fro');
            iterations = iterations + 1;
        end
        iters(rand_ind,n_ind) = iterations;
    end
end

figure
plot(n_v, mean(iters,1), 'LineWidth',1);
ylabel('expected iterations');
xlabel('dimensions');

