function [X] = DJGSC(S,T,nt,lambda,gamma)

tol = 1e-4;
maxIter = 500;
[d, n] = size(S);
[d, p] = size(T);
rho = 1.2;
max_mu = 1e20;
mu = 1e-7;
mu = 1.1;

%% Initializing optimization variables


X = zeros(n,p);
F = zeros(n,p);
B = zeros(n,p);
Y = zeros(n,p);
U = zeros(n,p);
k = n/nt;

%% Start main loop
iter = 0;
while iter<maxIter
    iter = iter + 1;

    for i = 0:k-1
        idx = nt*i+1:nt*i+nt;
        Fi = F(idx, :);
        Bi = B(idx, :);
        Xi = X(idx, :);
        Yi = Y(idx, :); 
        Ui = U(idx, :);
        all_indices = 1:size(F, 1); 
        remaining_idx = setdiff(idx,all_indices); 
        Xj = X(remaining_idx,:);
        Si = S(:,idx);
        Sj = S(:,remaining_idx);
        
        %update Fi
        temp = Xi - Yi/mu;
    
        Fi = solve_L21(temp,lambda/mu);
        
        %update Bi
        Bi = (mu*Xi-Ui)/(2*gamma*(Xj'*Xj)+mu*eye(p));
            
        %udpate Xi
        Xi = (2*(Si'*Si)+2*mu*eye(nt))\(2*Si'*(T-Sj*Xj)+mu*(Fi+Yi/mu+Bi+Ui/mu));


        F(idx, :) = Fi;
        B(idx, :) = Bi;
        X(idx, :) = Xi;

    end

    leq1 = F-X;
    leq2 = B-X;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));

    if (iter==1 || mod(iter,1)==0 || stopC<tol)
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',rank=' num2str(rank(X,1e-4*norm(X,2))) ',stopALM=' num2str(stopC,'%2.3e') ]);
    end
    if stopC<tol 
        break;
    else
        
        Y = Y + mu*leq1;
        U = U + mu*leq2;

        mu = min(max_mu,mu*rho);
    end
end
end

function [E] = solve_L21(M,lambda)
n = size(M,2);
E = M;
for i=1:n
    E(:,i) = solve_l2(M(:,i),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end

end



function [E] = solve_L12(M,lambda)
n = size(M,1);
E = M;
for i=1:n
    E(i,:) = solve_l1(M(i,:),lambda);
end
end

function [x] = solve_l1(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(1,length(w));
end

end



function [E] = solve_Q(M,lambda,nt)
n = size(M,2);

E = M;
for i=1:n
    E(:,i) = solve_l21(M(:,i),lambda,nt);
end
end

function [x] = solve_l21(w,lambda,nt)
% min lambda |x|_2 + |x-w|_2^2
m = size(w,1);
nw = 0;
for k = 1:m/nt
    nw = nw + norm(w((k-1)*nt+1:k*nt),2);
end
nw = nw^2;
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end

end


function [E] = solve_212(M,lambda,nt)
n = size(M,2);

E = M;
for i=1:n
    E(:,i) = solve_mix(M(:,i),lambda,nt);
end
end

function [x] = solve_mix(w,lambda,nt)
% min lambda |x|_2 + |x-w|_2^2
m = size(w,1);
nw = 0;
nwc = [];
x = [];
for k = 1:m/nt
    nwc = [nwc norm(w((k-1)*nt+1:k*nt),2)];
end
nw = norm(nwc,1);
for k = 1:m/nt
    if norm(w((k-1)*nt+1:k*nt))>0
        yk = w((k-1)*nt+1:k*nt);
        xk = yk-2*lambda*nw*yk/(m/nt*lambda*2+1)/nwc(1,k);
        x = [x;xk];
    else
        x = [x;zeros(nt,1)];
    end
end


end


