close all
clear

a = [1 2 3 2 1 0.1 0.1 0.1 0.1 0.1];
a = ones(1,10);
%a = [10, 1, 1, 1, 1, 1, 1, 1, 1, 1];
b = [0.1 0.1 0.1 0.1 0.1 1 2 3 2 1]';
b = [3 2 0.1 0.1 0.1 1 2 3 2 1]';
%a = [1 2 3 2 1 0 0 0 0 0];
%b = [0 0 0 0 0 1 2 3 2 1]';

a = b';

a = a/sum(a);
b = b/sum(b);
cost = zeros(10,10);
for i = 1:10
    for j = 1:10
        cost(i,j) = abs(i-j);
    end
end

figure;
plot(a,'blue');
hold on
p1 = plot(b,'red');
p1(1).LineWidth = 3;
    axis([0,10,0,0.4])
% pause
a
b

T = 100;

epsilon = 0.1;

delta = 0.005;

N=10;
fHist = ones(1,N);
gHist = ones(N,1);
aHist = a;

for (i=1:50)
    if false % true
        [plan, f, g] = Sinkhorn(cost, a, b, epsilon, 0.01);
%         f'
        fHist = [fHist; f];
        gHist = [gHist, g];
        F = epsilon*log(f)-T*log(a);
        Ffinite = isfinite(F);
        F(not(Ffinite)) = 0;
        F = F - Ffinite*sum(F)/sum(Ffinite);
        aNew = a-delta*F;
        if any(aNew < 0)
            Fpos = F;
            Fpos(Fpos<0) = 0;
            [deltaTemp, I] = min(a./Fpos);
            aNew = a-deltaTemp*F;
            aNew(I) = 0;
        end
        a = aNew;
%         a'
    else
        max_grad_descent_steps = 1;
        lambda = T;
        C = cost;
        K = exp(-cost/epsilon);
        source = b;
        target = a';
        
        source = reshape(source, length(source), 1);
        target = reshape(target, length(target), 1);

        opt_iters = 5000;
        
        [obj_val, target_OT] = OT_grad_descent(max_grad_descent_steps, lambda,...
            C, K, epsilon, source, target, opt_iters);
        a = target_OT';
%         a'
    end
    aHist = [aHist; a];
    plot(a,'blue');
    axis([0,10,0,1])
%     pause(0.1);
end
p2 = plot(a,'magenta');
p2(1).LineWidth = 3;
axis([0,10,0,1])