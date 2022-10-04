function L = ctranspose(L)
%
%  CTRANSPOSE The transpose of the psfMatrix matrix is
%             needed for matrix-vector multiply in iterative
%             restoration methods.  
%

%  J. Nagy 5/2/01

if L.transpose == 0
  L.transpose = 1;
else 
  L.transpose = 0;
end