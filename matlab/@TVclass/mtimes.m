function y = mtimes(L, x)

vec_flag=0;
if size(x,2)==1
    if L.transpose==0
        x=reshape(x,L.n,L.m);
    else
        x=reshape(x,2*L.n,L.m);
    end
    vec_flag=1;
end


if L.transpose==0
    [n,m]=size(x);
    y=zeros(2*n,m);
    xhat=fft2(x);
    y(1:n,:)=real(ifft2(L.L1.*xhat));
    y(n+1:end,:)=real(ifft2(L.L2.*xhat));
else
    [n,~]=size(x);
    xhat1=fft2(x(1:n/2,:));
    xhat2=fft2(x(n/2+1:end,:));
    y=real(ifft2(conj(L.L1).*xhat1+conj(L.L2).*xhat2));
end

if vec_flag==1
    y=y(:);
end
end
