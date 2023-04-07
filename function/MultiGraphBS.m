%----------------------------------------------------------------
% MultiGraph Band Selection
%----------------------------------------------------------------
function [index, Z, S, W, obj, score] = MultiGraphBS(X, alpha, lambda, c)
%% input:
%       dataset X (n times b, n for #samples, b for #bands)
%       hyper parameters alpha, gamma, sigma
%       the number of group: view_num
%       the label of group: view_label
%       the number of iteration: maxIter
%  output:
%       feature selection index: index
%       (for features) self-representation matrix: Z
%       (for samples) global similarity S
%       (for groups) weight matrix: W
% author: u 20200807
[n, d] = size(X);

%================================================
n_neighbors = n;%50;% Initialize similarity matrix for each graph
maxIter = 6;%number of iteration
%================================================

% Call the generate_graphs function in Python
Si = pyrunfile("generate_graphs.py", "Si", a = X, b = n_neighbors);
% Convert Python data type back to Matlab type
Si = double(Si);

graph_num=size(Si,3);%num of graph 
% Initialize input parameters
W = rand(graph_num, n);%ones(graph_num, n)./graph_num;
Z = rand(d,c);
S = rand(n,n);%sum(Si,3)/graph_num;%rand(n,n);
Yp = orth(rand(n,c));
obj = zeros(maxIter, 1);

 


% for i = 1:n
%     for j = 1:n
%         S(i,j) = exp(-(sum((X(i, :) - X(j, :)).^2))/sigma);
%     end
%     S(i,:) = S(i,:)./sum(S(i,:));
% end
% S=(S+S')./2;
 

%speed-up
XtX = X' * X;

for iter = 1:maxIter
    % update W
    for i = 1:n
        A_i = zeros(n, graph_num);
        for v = 1:graph_num
            A_i(:,v) = S(:,i)-Si(:,i,v);
        end
        part_bi = A_i'*A_i;
        part_1v = ones(graph_num,1);
        temp_inv = part_bi \ part_1v;
        W(:,i) = temp_inv / (part_1v' * temp_inv + 1e-15);
    end
    clear A_i part_bi part_1v;
    
    % update S
    for i = 1:n
        B_i = zeros(n, graph_num);
        for v = 1:graph_num
            B_i(:,v) = Si(:,i,v);
        end
        a_i = zeros(n, 1);
        for p = 1:n
            a_i(p) = norm(Yp(i,:)-Yp(p,:), 'fro')^2;
        end
        part_m = B_i * W(:,i);% - 0.25 * alpha * a_i;
        
        psi = zeros(n, 1);
        for j = 1:n
            temp = part_m - ones(n,n) * part_m / n + 0.5 * mean(psi);
            psi(j) = max(-2*temp(j), 0);
        end
        for j = 1:n
            temp = part_m - ones(n,n) * part_m / n + 0.5 * mean(psi);
            S(i, j) = max(temp(j), 0);
        end
    end
    clear B_i a_i part_m psi temp;
    
    % update Z
    L = diag(sum(S, 1)) - S;
    for loop = 1 : maxIter
        temp = 2 * sqrt(sum(Z.^2, 2)) + 1e-15;
        Q = diag(1./temp);
        temp1 = XtX + lambda * Q;
        Z = temp1 \ (X' * Yp);
    end
    clear temp Q temp1;

    % update Yp
    A=1+alpha.*L;
    B=X*Z;
    Yp=gpi(A,B,1);%refer to the implementation for paper "A generalized power iteration method
    %for solving quadratic problem on the Stiefel manifold", Feiping Nie, Rui Zhang, and Xuelong Li, 2017.

    
    % calculate objective function value1

    temp_formulation1 = 0;
    for i =1:n
        temp_S_j = zeros(n,1);
        for v = 1:graph_num
            temp_S_i = temp_S_j + W(v,i)*Si(:,i,v);
        end
        temp_formulation1 = temp_formulation1 + norm(S(:,i)-temp_S_i,'fro')^2;
    end
    temp_formulation2 = norm(Yp-X*Z,'fro')^2;
    temp_formulation3 = lambda * sum(sqrt(sum(Z.^2, 2)));%L2,1-norm
    temp_formulation4 = alpha * trace(Yp'*L * Yp);
    obj(iter) = temp_formulation1 + temp_formulation2 + temp_formulation3 + temp_formulation4;
    disp(obj(iter))
    clear temp_formulation1 temp_S_j temp_S_i temp_formulation2 temp_formulation3 temp_formulation4;

end

score=sum((Z.*Z),2);
[~,index]=sort(score,'descend');
end


%Title: A generalized power iteration method for solving quadratic problem on the Stiefel manifold
%% Authors: Feiping Nie, Rui Zhang, and Xuelong Li.
%Citation: SCIENCE CHINA Information Sciences 60, 112101 (2017); doi: 10.1007/s11432-016-9021-9
%View online: http://engine.scichina.com/doi/10.1007/s11432-016-9021-9
%View Table of Contents:http://engine.scichina.com/publisher/scp/journal/SCIS/60/11
%Published by the Science China Press
%% Generalized power iteration method (GPI) for solving min_{W��W=I}Tr(W��AW-2W^TB)
%Input: A as any symmetric matrix with dimension m*m; B as any skew matrix with dimension m*k,(m>=k);
%In particular, s can be chosen as 1 or 0, which stand for different ways of determining relaxation parameter alpha.
%i.e. 1 for the power method and 0 for the eigs function.
%Output: solution W and convergent curve.
function W=gpi(A,B,s)
if nargin<3
    s=1;
end
[m,k]=size(B);
if m<k
    disp('Warning: error input!!!');
    W=null(m,k);
    return;
end
A=max(A,A');

if s==0
    alpha=abs(eigs(A,1));
else if s==1
        ww=rand(m,1);
        for i=1:10
            m1=A*ww;
            q=m1./norm(m1,2);
            ww=q;
        end
        alpha=abs(ww'*A*ww);
else
    disp('Warning: error input!!!');
    W=null(m,k);
    return;
end
end

err1=1;t=1;
W=orth(rand(m,k));
% W_temp=eye(m,m);
% W = W_temp(:,1:k);

A_til=alpha.*eye(m)-A;
while t<5
    M=A_til*W+2*B;%2*A_til*W+2*B;
    [U,~,V]=svd(M,'econ');
    W=U*V';
    obj(t)=trace(W'*A*W-2.*W'*B);
    if t>=2
        err1=abs(obj(t-1)-obj(t));
    end
    t=t+1;
end
end




