import numpy as np
from scipy import sparse

from numba import njit

# Total variation operator (TV)
class TV:
    def __init__(self, n, m, transpose=False):
        self.n = n
        self.m = m
        self.transpose = transpose
        
        L1 = np.zeros((n, m))
        L1[0, :2] = np.array([-1, 1])
        L1 = np.fft.fft2(L1)

        L2 = np.zeros((n, m))
        L2[:2, 0] = np.array([-1, 1])
        L2 = np.fft.fft2(L2)

        self.L1 = L1
        self.L2 = L2

        if self.transpose == False:
            self.T = TV(self.n, self.m, transpose=True)

        self.eigs = (self.L1, self.L2)

    def __matmul__(self, x):
        if self.transpose:
            x = np.reshape(x, (2*self.n, self.m))
            xhat1 = np.fft.fft2(x[:self.n, :])
            xhat2 = np.fft.fft2(x[self.n:, :])
            y = np.real(np.fft.ifft2(self.L1.conj() * xhat1 + self.L2.conj() * xhat2))
        else:
            x = np.reshape(x, (self.n, self.m))
            y = np.zeros((2*self.n, self.m))
            xhat = np.fft.fft2(x)
            y[:self.n, :] = np.real(np.fft.ifft2(self.L1 * xhat))
            y[self.n:, :] = np.real(np.fft.ifft2(self.L2 * xhat))

        return y.flatten()


# Graph Laplacian Operator (L)
class GraphLaplacian():
    def __init__(self, I, sigmaInt, R=3):
        self.R = R - 1
        self.I = I
        self.sigmaInt = sigmaInt

        self.nr, self.nc = self.I.shape
        self.n = self.nr * self.nc

        self.L, self.W, self.D, self.A = self.computeL()


    def computeL(self):
        """
        Input:
        I: Image matrix
        sigmaInt: parameter of the weight function
        R: neighborhood radius in infinity norm
        """
        iW, jW, vW = utils_for_L(self.I, self.R, self.sigmaInt)

        W = sparse.csr_matrix((vW, (iW, jW)), (self.n, self.n))
        A = W
        W = W / sparse.linalg.norm(W, 'fro')

        d = W.sum()
        D = sparse.spdiags(d.T, 0, self.n, self.n)
        L = D - W

        return L, W, D, A


@njit()
def utils_for_L(I, R, sigmaInt):
    R = R-1
    nr, nc = I.shape
    n = nr * nc # n° nodes of the graph = n° of pixels

    k = 0
    iW = np.zeros(((2*R + 1) ** 2 * n, ))
    jW = np.zeros(((2*R + 1) ** 2 * n, ))
    vW = np.zeros(((2*R + 1) ** 2 * n, ))

    for x1 in range(nc):
        for y1 in range(nr):
            for x2 in range(max([x1-R, 0]), min([x1+R, nc])):
                for y2 in range(max([y1-R, 0]), min([y1+R, nr])):
                    node1 = y1 * nr + x1
                    node2 = y2 * nr + x2
                    if x1 != x2 or y1 != y2:
                        dist = I[x1, y1] - I[x2, y2]
                        iW[k] = node1
                        jW[k] = node2
                        vW[k] = np.exp(-dist**2/sigmaInt)
                        k = k+1
    iW = iW[:k-1]
    jW = jW[:k-1]
    vW = vW[:k-1]

    return iW, jW, vW