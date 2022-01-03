function targets = deterministic_dist(n,mesh_v)

    assert(n<=5);

    targets = zeros(n, length(mesh_v)^n);
    
    if n==1
        c = 1;
        for i1 = mesh_v
            targets(:,c)=i1;
            if norm(targets(:,c),1)>0
                targets(:,c) = targets(:,c)/norm(targets(:,c),1);
            else
                targets(:,c) = 1/n;
            end
            c = c + 1;
        end
    elseif n==2
        c = 1;
        for i1 = mesh_v
            for i2 = mesh_v
                targets(:,c)=[i1, i2]';
                if norm(targets(:,c),1)>0
                    targets(:,c) = targets(:,c)/norm(targets(:,c),1);
                else
                    targets(:,c) = 1/n;
                end
                c = c + 1;
            end
        end
    elseif n==3
        c = 1;
        for i1 = mesh_v
            for i2 = mesh_v
                for i3 = mesh_v
                    targets(:,c)=[i1, i2, i3]';
                    if norm(targets(:,c),1)>0
                        targets(:,c) = targets(:,c)/norm(targets(:,c),1);
                    else
                        targets(:,c) = 1/n;
                    end
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
                        if norm(targets(:,c),1)>0
                            targets(:,c) = targets(:,c)/norm(targets(:,c),1);
                        else
                            targets(:,c) = 1/n;
                        end
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
                            if norm(targets(:,c),1)>0
                                targets(:,c) = targets(:,c)/norm(targets(:,c),1);
                            else
                                targets(:,c) = 1/n;
                            end
                            c = c + 1;
                        end
                    end
                end
            end
        end
    end
    
    assert(all(isfinite(targets),'all'));
    