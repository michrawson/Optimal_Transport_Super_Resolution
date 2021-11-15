function target = rand_dist(n)

target = rand(n,1);
target = target/norm(target,1);
