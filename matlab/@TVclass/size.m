function s=size(L,j)
n=2*L.n*L.m;
m=L.n*L.m;
if nargin==1
    s=[n m];
else
    if j==1
        s=n;
    else
        s=m;
    end
end

