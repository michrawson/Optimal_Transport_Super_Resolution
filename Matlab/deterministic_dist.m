function targets = deterministic_dist(n,mesh_v)

    assert(n<=5);

    targets = zeros(n, length(mesh_v)^n);
    
    if n==1
        c = 1;
        for i1 = mesh_v
            targets(:,c)=i1;
            targets(:,c) = targets(:,c)/norm(targets(:,c),1);
            c = c + 1;
        end
    elseif n==2
        c = 1;
        for i1 = mesh_v
            for i2 = mesh_v
                targets(:,c)=[i1, i2]';
                targets(:,c) = targets(:,c)/norm(targets(:,c),1);
                c = c + 1;
            end
        end
    elseif n==3
        c = 1;
        for i1 = mesh_v
            for i2 = mesh_v
                for i3 = mesh_v
                    targets(:,c)=[i1, i2, i3]';
                    targets(:,c) = targets(:,c)/norm(targets(:,c),1);
                    c = c + 1;
                end
            end
        end
    elseif n==4
        c = 1;
        for i1 = mesh_v
            for i2 = mesh_v
                for i3 = mesh_v
                    for i4 = mesh_v
                        targets(:,c)=[i1, i2, i3, i4]';
                        targets(:,c) = targets(:,c)/norm(targets(:,c),1);
                        c = c + 1;
                    end
                end
            end
        end
    elseif n==5
        c = 1;
        for i1 = mesh_v
            for i2 = mesh_v
                for i3 = mesh_v
                    for i4 = mesh_v
                        for i5 = mesh_v
                            targets(:,c)=[i1, i2, i3, i4, i5]';
                            targets(:,c) = targets(:,c)/norm(targets(:,c),1);
                            c = c + 1;
                        end
                    end
                end
            end
        end
    end
    