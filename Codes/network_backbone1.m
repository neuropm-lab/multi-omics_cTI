function [BackBone, ne_add] = network_backbone1(Matrix, min_dens_nc)
% Yasser Iturria & Marlis Ontivero, 27 Oct, 2010.
% Modify by Ontivero-Ortega Marlis, 8, May, 2012. 

nnode = size(Matrix, 1);
Matrix(diag(true(nnode, 1))) = 0; % Eliminar conexiones de la diagonal de   
                                  % la matrix.

% Minimo numero de aristas (ponderadas por el peso) que hacen que el grafo
% sea totalmente conexo.
L = tril(Matrix, -1);
L = sparse(1 ./ L);
Tree = graphminspantree(sparse(L));
Tree(Tree > 0) = Matrix(Tree > 0);
BackBone = full(Tree + Tree');
% figure; imagesc(BackBone);

node_dens_nc = sum(BackBone > 0, 2);
mindens = min(node_dens_nc);
if mindens < min_dens_nc
    matew = Matrix;
    matew(BackBone > 0) = 0;
    ne = length(find(matew)) / 2;
       
    ne_add = 0;
    while (mindens < min_dens_nc) && (ne_add < ne)
        inds = find(node_dens_nc == mindens);
        T = matew(inds, :);
        maxc = max(T(:));
        indsmax = find(T == maxc);
        indsmax = indsmax(1);
        [r1, c1] = ind2sub(size(T), indsmax);
        
        BackBone(inds(r1), c1) = maxc;
        BackBone(c1, inds(r1)) = maxc;
        
        matew(inds(r1), c1) = 0;
        matew(c1, inds(r1)) = 0;
        
        node_dens_nc(inds(r1)) = node_dens_nc(inds(r1)) + 1;
        node_dens_nc(c1) = node_dens_nc(c1) + 1;
        
        ne_add = ne_add + 1;
        mindens = min(node_dens_nc);
    end  
end

% ew_inds = find(matew);
% ew = matew(ew_inds);
% [ew_ord, ew_inds_ord] = sort(ew, 'descend');
% ew_inds = ew_inds(ew_inds_ord);
% [x, y] = ind2sub(size(Matrix), ew_inds);
% for node = 1 : length(node_dens_nc)
%     while node_dens_nc(node) < min_dens_nc
%         values = sort(Matrix(node,:),'descend');
%         BackBone(node,find(Matrix(node,:) == values(node_dens_nc(node)+1))) = values(node_dens_nc(node)+1);
%         BackBone(find(Matrix(:,node) == values(node_dens_nc(node)+1)),node) = values(node_dens_nc(node)+1);
%         node_dens_nc(node) = node_dens_nc(node) + 1;
%     end
% end
    
    
    
% while (mindens <  min_dens_nc) && (ne_add < length(ew_ord))  
%     BackBone(x(ne_add), y(ne_add)) = ew_ord(ne_add);
%     BackBone(y(ne_add), x(ne_add)) = ew_ord(ne_add);
%     
%     node_dens_nc(x(ne_add)) = node_dens_nc(x(ne_add)) + 1;
%     node_dens_nc(y(ne_add)) = node_dens_nc(y(ne_add)) + 1;
%     
%     ne_add = ne_add + 1;
%     mindens = min(node_dens_nc);
% end
% 
% ne_add = ne_add - 1;

%figure; imagesc(BackBone);

end



