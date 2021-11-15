function y = num_peaks(x)

y = 0;

i = 1;
if x(i) > x(i+1) 
    y = y + 1;
end

for i = 2:length(x)-1
    if x(i-1) < x(i) && x(i) > x(i+1) 
        y = y + 1;
    end
end

i = length(x);
if x(i-1) < x(i)
    y = y + 1;
end
