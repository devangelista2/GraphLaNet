import numpy as np
import scipy
from scipy.stats import ortho_group

def cs_decomp(Q1, Q2):
    m1, n1 = Q1.shape
    m2, n2 = Q2.shape
    
    assert n1 == n2
    n = n1

    U1, C0, V1 = np.linalg.svd(Q1)
    print(C0)

n = 128

Q = ortho_group.rvs(dim=512)
Q = Q[:, :n]


m1 = 64
m2 = 8

Q1 = Q[:m1, :]
Q2 = Q[m1:, :]

cs_decomp(Q1, Q2)