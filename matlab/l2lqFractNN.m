function x=l2lqFractNN(A,b,options)
% This function solves the minimization problem
% x=argmin 1/2*||Ax-b||_2^2+mu/q*||LG^alpha*x||_q^q
% 
% where x is the vectorization of a n x m image, LG is the graph 
% laplacian constructed from an approximation of x, and mu is determined by 
% the discrepancy principle (if it is not given)
% 
% Input:
% A: matrix of the system (it can be an object such that A*x computes the
%    Mvp with A and A'*x computes the Mvp with A');
% b: right-hand side
% options: structure containing the data required by the algorithm
% options.L: regularization operator for the computation of the initial
%            approximation of x (if not defined or empty L=TV);
% options.q: value of q (if not defined or empty q=0.1);
% options.epsilon: smoothing parameter for the minimization algorithm 
%                  (if not defined or empty epsilon=1e-2/max(b(:)));
% options.mu: value or the regularization parameter (must be given if
%             noise_norm is either empty or not provided);
% options.noise_norm: norm of the noise in the data (must be given if
%                     mu is either empty or not provided);
% options.tau: parameter of the discrepancy principle (if not defined or
%              empty tau=1.01);
% options.iter: number of iteration for the minimization algorithm (if not 
%               defined or empty iter=200);
% options.tol: tollerance for the stopping criterion of the minimization 
%              algorithm (if not defined or empty tol=1e-4);
% options.Rest: parameter for the restarting of the minimization algorithm 
%               (if not defined or empty Rest=30);
% options.n: size of the image (if not defined or empty n=sqrt(numel(b)));
% options.m: size of the image (if not defined or empty m=sqrt(numel(b)));
% options.sigmaInt: parameter of the graph laplacian (if not defined or 
%                   empty sigmaInt=1e-3);
% options.R: parameter of the graph laplacian (if not defined or 
%                   empty R=10).
% options.alpha: exponent of the graph laplacian (if not defined or empty 
%                alpha=1);

% Check input
if ~isfield(options,'n') || isempty(options.n)
    n=sqrt(numel(b));
else
    n=options.n;
end
if ~isfield(options,'m') || isempty(options.m)
    m=sqrt(numel(b));
else
    m=options.m;
end
if ~isfield(options,'recon') || isempty(options.recon)
    recon=zeros(size(b));
else
    recon=options.recon;
end
if ~isfield(options,'q') || isempty(options.q)
    q=0.1;
else
    q=options.q;
end
if ~isfield(options,'epsilon') || isempty(options.epsilon)
    epsilon=1e-2/max(b(:));
else
    epsilon=options.epsilon;
end
if ~isfield(options,'tau') || isempty(options.tau)
    tau=1.01;
else
    tau=options.tau;
end
if ~isfield(options,'iter') || isempty(options.iter)
    iter=200;
else
    iter=options.iter;
end
if ~isfield(options,'tol') || isempty(options.tol)
    tol=1e-4;
else
    tol=options.tol;
end
if ~isfield(options,'Rest') || isempty(options.Rest)
    Rest=30;
else
    Rest=options.Rest;
end
if ~isfield(options,'sigmaInt') || isempty(options.sigmaInt)
    sigmaInt=1e-3;
else
    sigmaInt=options.sigmaInt;
end
if ~isfield(options,'R') || isempty(options.R)
    R=10;
else
    R=options.R;
end
if ~isfield(options,'alpha') || isempty(options.alpha)
    alpha=1;
else
    alpha=options.alpha;
end

if ~isfield(options,'noise_norm') || isempty(options.noise_norm)
    noise_norm=[];
else
    noise_norm=options.noise_norm;
end
if ~isfield(options,'mu') || isempty(options.mu)
    mu=[];
else
    mu=options.mu;
end
if isempty(noise_norm) && isempty(mu)
    error('At least one of the fields options.mu and options.noise_norm must be given');
end
if ~isempty(noise_norm) && ~isempty(mu)
    warning(['Both options.noise_norm and options.mu were defined, the value of mu will be fixed at ',num2str(mu)])
end


% waitbar
h=waitbar(0,'l_2-l_q Fractional - Computations in progress...');

% Initializations
delta=tau*noise_norm;
b=b(:);
x=A'*b;

% Creating initial space
v=x;
nv=norm(v(:));
V=v(:)/nv;
AV=A*V;

% Creating graph laplacian from an initial approximation

L = computeL(recon,sigmaInt,R);

% Compute L^alpha*v using Lanczos
d=10;
LV=computeLv(L,alpha,V,d);

% Inital QR factorizations
[QA,RA]=qr(AV,0);
[QL,RL]=qr(LV,0);

% Initial weights
u=L*x;
y=nv;
% Begin MM iterations
for k=1:iter
    if mod(k,Rest)==0
        % Restarting the Krylov subspace to save memory
        x=V*y;
        clear V AV LV QA RA QR RL
        V=x/norm(x);
        AV=A*V;
        LV=computeLv(L,alpha,V,d);
        [QA,RA]=qr(AV,0);
        [QL,RL]=qr(LV,0);
    end
    % Store previous iteration for stopping criterion
    y_old=y;
    
    % Compute weights for approximating the q norm with the 2 norm
    wr=u(:).*(1-((u(:).^2+epsilon^2)/epsilon^2).^(q/2-1));
    
    % Solve re-weighted linear system selectin the parameter with the DP
    c=epsilon^(q-2);
    if isempty(mu)
        eta=discrepancyPrinciple(delta,RA,RL,QA,QL,b,wr,c);
    else
        eta=c*mu;
    end
    y=[RA; sqrt(eta)*RL]\[QA'*(b(:)); sqrt(eta)*(QL'*(wr))];

    % Check stopping criteria
    if k>1 && numel(y)>1 && norm(y-[y_old;0],'fro')/norm([y_old;0],'fro')<tol
        break
    end

    if k<iter && mod(k+1,Rest)~=0
        % Enlarge the space and update QR factorizations
        v=AV*y-b;
        u=LV*y;
        ra=v;
        ra=A'*ra;
        rb=(u-wr(:));
        rb=L'*rb;
        r=ra(:)+eta*rb(:);
        r=r-V*(V'*r);
        r=r-V*(V'*r);
        [AV,LV,QA,RA,QL,RL,V]=updateQR(A,L,AV,LV,QA,RA,QL,RL,V,r,alpha,d);
    end
    waitbar(k/(iter-1));
end
x= reshape(V*y,n,m);
try
    close(h)
catch
    warning('Waitbar not closed')
end
end


function mu=discrepancyPrinciple(delta,RA,RL,QA,QL,b,wr,c)
mu=1e-30;
[U,V,~,C,S]=gsvd(RA,RL);
what=V'*QL'*wr;
bhat=QA'*b;
bb=b-QA*bhat;
nrmbb=norm(bb);
bhat=U'*bhat;
a=diag(C);
l=diag(S);
for i=1:30
    mu_old=mu;
    f=((c*a.*l.*what-c*bhat.*l.^2).^2)'*((mu*a.^2+c*l.^2).^-2)-delta^2+nrmbb^2;
    fprime=-2*(a.^2.*(c*a.*l.*what-c*bhat.*l.^2).^2)'*((mu*a.^2+c*l.^2).^-3);
    mu=mu-f/fprime;
    if abs(mu_old-mu)/mu_old<1e-6
        break
    end
end
mu=c/mu;
end

function [AV,LV,QA,RA,QL,RL,V]=updateQR(A,L,AV,LV,QA,RA,QL,RL,V,r,alpha,d)
vn=r/norm(r(:));
Avn=A*vn;
AV=[AV, Avn(:)];
% Lvn=L*vn;
Lvn=computeLv(L,alpha,vn,d);
LV=[LV, Lvn(:)];
V=[V,vn];
rA=QA'*Avn(:);
qA=Avn(:)-QA*rA;
tA=norm(qA(:));
qtA=qA/tA;
QA=[QA,qtA];
RA=[RA rA;zeros(1,length(rA)) tA];
rL=QL'*Lvn(:);
qL=Lvn(:)-QL*rL;
tL=norm(qL(:));
qtL=qL/tL;
QL=[QL,qtL];
RL=[RL rL;zeros(1,length(rL)) tL];
end

function Lv=computeLv(LG,alpha,V,d)
Lv=zeros(size(V));
for j=1:size(V,2)
    v=V(:,j);
    switch alpha
        case 1
            Lv(:,j)=LG*v;
        case 0
            Lv(:,j)=v;
        otherwise
            [T,W]=lanczos(LG,v,d);
            [Q,La]=eig(T);
            la=diag(La);
            Lv(:,j)=W*(Q*(la.^alpha.*(Q'*(W'*v))));
    end
end
end

function [T,V]=lanczos(A,v,d)
V=zeros(size(A,1),d);
T=zeros(d,d);

v=v/norm(v);
V(:,1)=v;
w=A*v;
alpha=w'*v;
w=w-alpha*v;
T(1,1)=alpha;
for k=2:d
    beta=norm(w);
    T(k-1,k)=beta;
    T(k,k-1)=beta;
    v=w/beta;
    V(:,k)=v;
    w=A*v;
    alpha=w'*v;
    T(k, k)=alpha;
    if k~=d
        w=w-alpha*v-beta*V(:,k-1);
    end
end

end

function [x,mu]=KTikhonovGenGCV(A,b,k,L)
% Solves the Tikhonov problem in general form
% x=argmin ||Ax-b||^2+mu*||Lx||
% in the GK Krylov subspace of dimension k 
% Determining mu with the GCV

[~,B,V] = lanc_b(A,b(:),k);
e=zeros(2*k+1,1);
e(1)=norm(b(:));
lv=L*V(:,1);
LV=zeros(length(lv),k);
LV(:,1)=lv;
for j=1:k
    LV(:,j)=L*V(:,j);
end
[~,R]=qr(LV,0);

mu=gcv(full(B),R,e(1:k+1));

y=[B;sqrt(mu)*R]\e;

x=V*y;
end

function mu=gcv(A,L,b)
[U,~,~,S,La] = gsvd(A,L);
bhat=U'*b;
l=diag(La);
s=diag(S);
extreme=1;
M=1e2;
while extreme
    mu=fminbnd(@gcv_funct,0,M,[],s,l,bhat(1:length(s)));
    if abs(mu-M)/M<1e-3
        M=M*100;
    else
        extreme=0;
    end
    if M>1e10
        extreme=0;
    end
end
end

function G=gcv_funct(mu,s,l,bhat)
num=(l.^2.*bhat./(s.^2+mu*l.^2)).^2;
num=sum(num);
den=(l.^2./(s.^2+mu*l.^2));
den=sum(den)^2;
G=num/den;
end


function [U,B,V] = lanc_b(A,p,k)
N=numel(p);
M=numel(A'*p);
U = zeros(N,k+1);
V = zeros(M,k);
B = sparse(k+1,k);
% Prepare for Lanczos iteration.
v = zeros(M,1);
beta = norm(p);
if (beta==0)
    error('Starting vector must be nonzero')
end
u = p/beta;
U(:,1) = u;
% Perform Lanczos bidiagonalization with/without reorthogonalization.
for i=1:k
    r=A'*u;
    r = r - beta*v;
    for j=1:i-1
        r = r - (V(:,j)'*r)*V(:,j);
    end
    alpha = norm(r);
    v = r/alpha;
    B(i,i) = alpha;
    V(:,i) = v;
    p=A*v;
    p = p - alpha*u;
    for j=1:i
        p = p - (U(:,j)'*p)*U(:,j);
    end
    beta = norm(p);
    u = p/beta;
    B(i+1,i) = beta;
    U(:,i+1) = u;
end

end

function [L,W,D,A] = computeL(I,sigmaInt,R)
% Input:  - I ,image matrix  
%         - sigmaInt, parameter of the weight function
%         - R, neighborhood radius in infinity norm

if nargin<3
    R=3;
end
R=R-1;
[nr,nc]=size(I);
n=nr*nc; %n° nodes of the graph = n° of pixels
% vecI=I(:); %gives an order to the pixels


k=1; iW=zeros((2*R+1)^2*n,1); jW=zeros((2*R+1)^2*n,1); vW=zeros((2*R+1)^2*n,1);
for x1=1:nc
for y1=1:nr
    for x2=max(x1-R,1):min(x1+R,nc)
    for y2=max(y1-R,1):min(y1+R,nr)
        node1=(y1-1)*nr+x1; node2=(y2-1)*nr+x2; %sorting of the pixels
        if x1~=x2 || y1~=y2
            dist=I(x1,y1)-I(x2,y2);
            iW(k)=node1; jW(k)=node2; 
            vW(k)=exp(-dist^2/sigmaInt);
            k=k+1;
        end
    end
    end
end
end
iW=iW(1:k-1); jW=jW(1:k-1); vW=vW(1:k-1);
W=sparse(iW,jW,vW,n,n);
A=W;
W=W./norm(W(:));

d=sum(W);
D=spdiags(d',0,n,n);
L=D-W;

end