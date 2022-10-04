function L=TVclass(n,m)
L1=[-1 1];
L1=padarray(L1,[n,m]-size(L1),0,'post');
L1=fft2(circshift(L1,1-[1,1]));

L2=[-1; 1];
L2=padarray(L2,[n,m]-size(L2),0,'post');
L2=fft2(circshift(L2,1-[1,1]));

L.L1=L1;
L.L2=L2;

L.transpose=0;

L.n=n;
L.m=m;

L=class(L,'TVclass');
end