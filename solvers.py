# Libraries
import scipy
from scipy import optimize, sparse
import numpy as np
import operators

import pygsvd

from numba import jit, njit

def l2lq(A, b, L='TV', q=0.1, epsilon=None, mu=None, noise_norm=None, tau=1.01,
        maxit=200, tol=1e-4, rest=30, n=None, m=None, sigmaInt=1e-3, R=10, alpha=1):
    # This function solves the minimization problem 
    # 
    # x=argmin 1/2*||Ax-b||_2^2+mu/q*||LG^alpha*x||_q^q
    #
    # where x is the vectorization of a n x m image, LG is the graph laplacian constructed from an approximation of x, and mu is determined by 
    # the discrepancy principle (if it is not given)

    """
    Input:

    A: an object that supports the operations A@x and A.T@x
    b: right-hand side
    L: regularization operator for the computation of the approximation x_0 of x, required for Laplacian Graph.
       By default, L = TV.
    q: value of q. By default, q = 0.1
    epsilon: smoothing parameter for || ||_q. By default, epsilon = 1e-2 / max(b).
    mu: regularization parameter. Must be given if noise_norm is None.
    noise_norm: Norm of the noise. Must be given if mu is None.
    tau: parameter for discrepancy principle. By default, tau = 1.01.
    maxit: stopping criteria. By default, maxit = 200.
    tol: tollerance stopping criteria. By default, tol = 1e-4.
    rest: Iteration before restarting of minimization algorithm. By default, rest = 30.
    m, n: size of the image. By default they are both int(sqrt(b.shape)). 
    sigmaInt: parameter of the GraphLaplacian. By default, sigmaInt = 1e-3.
    R: parameter of the GraphLaplacian. By default, R = 10.
    alpha: exponent of GraphLaplacian. By default, alpha=1.
    """

    # Define special default parameter
    if n is None:
        n = int(np.sqrt(len(b.flatten())))
    if m is None:
        m = len(b.flatten()) // n
    
    if epsilon is None:
        epsilon = 1e-2 / b.max()
    
    if mu is None and noise_norm is None:
        raise ValueError

    if L == 'TV':
        L = operators.TV(n, m)
    
    # Initialization
    if noise_norm is not None:
        delta = tau * noise_norm
    b = b.flatten() # Vectorize b
    x = A.T @ b

    k = 0

    # Creating initial space (setup for bidiagonalization procedure)
    v = x
    nv = np.linalg.norm(v)
    V = v / nv # TODO: Why V is the normalization of x? (line 12 of the pseudocode)
    AV = A@V

    # Creating GraphLaplacian
    l = 20
    xGCV = KTikhonovGenGCV(A, b, l, L) # Line 5-8 of the pseudo-code
    L = computeL(xGCV.reshape((m, n)), sigmaInt, R) # Line 9-11 of the pseudo-code

    # Computing L^alpha*v using Lanczos
    d = 10
    LV = computeLV(L, alpha, V, d) # Here we are just multiplying L@V (since alpha=1)
                                   # V is a vector -> LV is a vector

    # Initial QR factorization
    QA, RA, _ = np.linalg.qr(AV, 'reduced')  # AV and LV are (n, 1) vectors ->
    QL, RL, _ = np.linalg.qr(LV, 'reduced')  #   QA, QL are (n, n) matrices,
                                             #   RA, RL are (n, 1) vectors with only
                                             #   one non-zero element (the uppermost)
    # Initial weights
    u = L@x # For k=0, u0 = LV0 y0 = LV0 V0^T x = L x
    y = nv # Since x = V nv and V^T V = 1, then V^T x = V^T V nv = nv

    # Begin MM iterations
    CONTINUE = True
    while CONTINUE:
        if k % rest == 0:
            # Restarting the Krylov subspace to save memory
            x = V@y 

            del V, AV, LV, QA, RA, QL, RL
            V = x / np.linalg.norm(x)
            AV = A @ V
            LV = computeLV(L, alpha, V, d)

            QA, RA = np.linalg.qr(AV, 'reduced')
            QL, RL = np.linalg.qr(LV, 'reduced')
        # Store old iteration for stopping criteria
        y_old = y

        # Compute weights for approximating the q norm with the 2-norm
        wr = u * (1 - ((u**2 + epsilon**2) / epsilon**2) ** (q/2 - 1)) # line 18
                                                    # is this an element-wise product?

        # Solve the re-weighted linear system selecting the parameters with DP
        c = epsilon ** (q-2)
        if mu is None:
            eta = discrepancyPrinciple(delta, RA, RL, QA, QL, b, wr, c) # line 20
        else:
            eta = mu * c # line 19 of the pseudocode
        # compute y line 21 of the pseudocode
        y = np.linalg.solve(np.concatenate([RA, np.sqrt(eta)*RL], axis=1), np.concatenate([QA.T@b, np.sqrt(eta)*(QL.T@wr)], axis=1))

        # Check stopping criteria
        CONTINUE = np.linalg.norm(y - np.concatenate([y_old, np.zeros((1, ))], axis=0)) / np.linalg.norm(np.concatenate([y_old, np.zeros((1, ))], axis=0)) > tol and k < maxit

        if k < maxit and (k+1) % rest != 0:
            # Enlarge the space and update QR factorization
            v = AV @ y - b # AV is a vector -> AV @ y is a scalar. ???? No corr. in paper
            u = LV @ y     # LV is a vecotr -> LV @ y is a scalar. ????
            ra = v         # vector
            ra = A.T @ ra  # vector -> ra = A^T (AV y - b)
            rb = (u - wr)  # vector
            rb = L.T @ rb  # L^T (LVy - wr)
            r = ra + eta*r # Line 22 
            r = r - V@(V.T@r) # Line 23
            r = r - V@(V.T@r) # why twice??
            AV, LV, QA, RA, QL, RL, V = updateQR(A, L, AV, LV, QA, RA, QL, RL, V, r, alpha, d)
        
        # Update step
        k = k + 1

        # Exit
        return np.reshape(V@y, (n, m))



def computeLV(LG, alpha, V, d):
    Lv = np.zeros(V.shape)
    for j in range(V.shape[1]):
        v = V[:, j]

        if alpha == 1:
            Lv[:, j] = LG @ v
        elif alpha == 0:
            Lv[:, j] = v
        else:
            T, W = lanczos(LG, v, d)
            Q, La = np.linalg.eig(T)
            la = np.diag(La)
            Lv[:, j] = W @ (Q @ (la ** alpha * (Q.T @ (W.T @ v))))

    return Lv

def lanczos(A, v, d):
    V = np.zeros((A.shape[0], d))
    T = np.zeros((d, d))

    v = v / np.linalg.norm(v)
    V[:, 0] = v
    w = A@V
    alpha = w.T @ v
    w = w - alpha@V
    T[0, 0] = alpha
    for k in range(1, d):
        beta = np.linalg.norm(w)
        T[k-1, k] = beta
        T[k, k-1] = beta
        v = w / beta
        V[:, k] = V
        w = A@V
        alpha = w.T @ v
        T[k, k] = alpha

        if k != d:
            w = w - alpha * v - beta * V[:, k-1]

    return T, V


def discrepancyPrinciple(delta, RA, RL, QA, QL, b, wr, c):
    mu = 1e-30
    U, V, _, C, S = gsvd(RA, RL)
    what = V.T @ (QL.T @ wr)
    bhat = QA.T @ b
    bb = b - QA @ bhat
    nrmbb = np.linalg.norm(bb)
    bhat = U.T @ bhat
    a = np.diag(C)
    l = np.diag(S)
    for i in range(30):
        mu_old = mu
        f1 = ((c * a * l * what - c * bhat * l ** 2) ** 2)
        f2 = ((mu * a ** 2 + c * l ** 2) ** (-2)) . delta ** 2 + nrmbb ** 2
        f = f1.T @ f2
        fprime1 = -2 * (a ** 2 * (c * a * l * what - c * bhat * l**2) ** 2)
        fprime2 = ((mu * a ** 2 + c * l ** 2) ** (-3))
        fprime = fprime1.T * fprime2

        mu = mu - f / fprime
        if np.abs(mu_old - mu) / mu_old < 1e-6:
            break
    mu = c / mu
    return mu


def updateQR(A, L, AV, LV, QA, RA, QL, RL, V, r, alpha, d):
    vn = r / np.linalg.norm(r) # Line 24
    Avn = A @ vn               # Avnew = A r / ||r||
    AV = np.concatenate([AV, Avn], axis=0) # AV_k+1 = [AV_k, vnew]
    
    Lvn = computeLV(L, alpha, vn, d)       # L vnew
    LV = np.concatenate([V, vn], axis=0)   # LV_k+1 = [LV_k, vnew]

    # ??? How does QR update works for A
    rA = QA.T @ Avn
    qA = Avn - QA @ rA
    ta = np.linalg.norm(qA)
    qtA = qA / ta
    QA = np.concatenate([QA, qtA], axis=0)
    RA = np.concatenate([np.concatenate([RA, rA], axis=0),
                         np.concatenate([np.zeros((1, len(rA))), ta], axis=0)], axis=1)
    
    # Same as QR update for A
    rL = QL.T @ Lvn
    qL = Lvn - QL @ rL
    tL = np.linalg.norm(qL)
    qtL = qL / tL
    QL = np.concatenate([QL, qtL], axis=0)
    RL = np.concatenate([np.concatenate([RL, rL], axis=0), 
                         np.concatenate([np.zeros((1, len(rL))), tL], axis=0)], axis=1)
    return AV, LV, QA, RA, QL, RL, V

def KTikhonovGenGCV(A, b, k, L):
    # Solves the Tikhonov problem in general form 
    # x = argmin || Ax - b ||^2 + mu * ||Lx||
    # in the GK Krylov subspace of dimension k, determinining mu with the GCV

    _, B, V = lanc_b(A, b, k)
    e = np.zeros((2*k+1, 1))
    e[0] = np.linalg.norm(b)
    lv = L @ V[:, 0]
    LV = np.zeros((len(lv), k))
    LV[:, 0] = lv
    for j in range(k):
        LV[:, j] = L @ V[:, j]
    
    _, R = np.linalg.qr(LV)
    
    mu = gcv(B, R, e[:k+1])
    y = np.linalg.solve(np.concatenate([B, np.sqrt(mu) * R], axis=1), e)
    x = V @ y
    return x, mu

def gcv(A, L, b):
    U, _, _, S, La = gsvd(A, L)
    bhat = U.T @ b
    l = np.diag(La)
    s = np.diag(S)
    extreme = True
    M = 1e2

    while extreme:
        mu = optimize.fminbound(gcv_funct, 0, M, s, l, bhat[:len(s)])
        if np.abs(mu - M) / M < 1e-3:
            M = M * 100
        else:
            extreme = False
        
        if M > 1e10:
            extreme = False
    
    return mu

def gcv_funct(mu, s, l, bhat):
    num = (l ** 2 * bhat / (s**2 + mu * l ** 2)) ** 2
    num = np.sum(num)
    den = (l ** 2 / (s**2 + mu * l ** 2))
    den = np.sum(den) ** 2
    G = num / den
    return G

def gsvd(A, B):
    C, S, X, U, V = pygsvd.gsvd(A, B)
    return U, V, X, C, S

def lanc_b(A, p, k):
    N = np.size(p)
    M = np.size(A.T@p)

    U = np.zeros((N, k+1))
    V = np.zeros((M, k))
    B = sparse.bsr_matrix((k+1, k))

    # Prepare for Lanczos iteration
    v = np.zeros((M, 1))
    beta = np.linalg.norm(p)

    if beta == 0:
        raise ValueError
    
    u = p / beta
    U[:, 0] = u

    # Perform Lanczos bidiagonalization with/without reorthogonalization
    for i in range(k):
        r = A.T @ u
        r = r - beta * v
        for j in range(i-1):
            r = r - (V[:, j].T @ r) * V[:, j]
        
        alpha = np.linalg.norm(r)
        v = r / alpha 
        B[i, i] = alpha
        V[:, i] = v
        p = A @ v
        p = p - alpha * u
        for j in range(i):
            p = p - (U[:, j].T @ p) * U[:, j]
        beta = np.linalg.norm(p)
        u = p / beta
        B[i+1, i] = beta
        U[:, i+1] = u

    return U, B, V