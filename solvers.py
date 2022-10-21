import numpy as np

import scipy
from scipy import optimize, sparse, linalg

import operators
import solvers
import utils

def GraphLaTV(A, b, q=0.1, epsilon=None, mu=None, maxit=200, tol=1e-4, rest=30, n=None, m=None, sigmaInt=1e-3, R=10):
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
    q: value of q. By default, q = 0.1
    epsilon: smoothing parameter for || ||_q. By default, epsilon = 1e-2 / max(b).
    mu: regularization parameter. Must be given if noise_norm is None.
    maxit: stopping criteria. By default, maxit = 200.
    tol: tollerance stopping criteria. By default, tol = 1e-4.
    rest: Iteration before restarting of minimization algorithm. By default, rest = 30.
    m, n: size of the image. By default they are both int(sqrt(b.shape)). 
    sigmaInt: parameter of the GraphLaplacian. By default, sigmaInt = 1e-3.
    R: parameter of the GraphLaplacian. By default, R = 10.
    """

    # Define special default parameter
    if n is None:
        n = int(np.sqrt(len(b.flatten())))
    if m is None:
        m = len(b.flatten()) // n
    
    if epsilon is None:
        epsilon = 1e-2 / b.max()
    
    # Initialization
    L = operators.TV(n, m)
    
    b = np.expand_dims(b.flatten(), 1) # Vectorize b
    x = A.T @ b

    k = 0

    # Creating initial space (setup for bidiagonalization procedure)
    v = x 
    nv = np.linalg.norm(v)
    V = v / nv
    AV = A@V

    # Creating GraphLaplacian
    print("Creating Graph Laplacian")
    l = 20
    xGCV = utils.KTikhonovGenGCV(A, b, l, L) # Line 5-8 of the pseudo-code
    L = operators.GraphLaplacian(xGCV.reshape((256, 256)), sigmaInt, R).L # Line 9-11 of the pseudo-code
    print("Done!")
    print("")

    # Computing L^alpha*v using Lanczos
    LV = L @ V # Here we are just multiplying L@V (since alpha=1)

    # Initial QR factorization
    QA, RA = np.linalg.qr(AV, mode='reduced')  
    QL, RL = np.linalg.qr(LV, mode='reduced')  

    # Initial weights
    u = L@x
    y = np.array([[nv]])

    # Begin MM iterations
    CONTINUE = True
    while CONTINUE:
        if k % rest == 0:
            # Restarting the Krylov subspace to save memory
            x = V@y 

            del V, AV, LV, QA, RA, QL, RL
            V = x / np.linalg.norm(x)
            AV = A @ V
            LV = L @ V

            QA, RA = np.linalg.qr(AV, 'reduced')
            QL, RL = np.linalg.qr(LV, 'reduced')

        # Store old iteration for stopping criteria
        y_old = y

        # Compute weights for approximating the q norm with the 2-norm
        wr = u * (1 - ((u**2 + epsilon**2) / epsilon**2) ** (q/2 - 1)) # line 18

        # Solve the re-weighted linear system selecting the parameters with DP
        c = epsilon ** (q-2)
        eta = mu * c # line 19 of the pseudocode

        # compute y line 21 of the pseudocode
        RARL = np.concatenate([RA, np.sqrt(eta)*RL], axis=0)
        QAQL = np.concatenate([QA.T@b, np.sqrt(eta)*(QL.T@wr)], axis=0)
        y, _, _, _ = np.linalg.lstsq(RARL, QAQL, rcond=-1)

        # Check stopping criteria
        CONTINUE = np.linalg.norm(y - np.concatenate([y_old, np.zeros((1, 1))], axis=0)) / np.linalg.norm(np.concatenate([y_old, np.zeros((1, 1))], axis=0)) > tol and k < maxit
        if CONTINUE == False:
            break

        if k < maxit and (k+1) % rest != 0:
            # Enlarge the space and update QR factorization
            v = AV @ y - b 
            u = LV @ y    
            ra = A.T @ v           
            rb = L.T @ (u - wr) 
            r = ra + eta * rb # Line 22 
            r = r - V@(V.T@r) # Line 23
            AV, LV, QA, RA, QL, RL, V = utils.updateQR(A, L, AV, LV, QA, RA, QL, RL, V, r)
        
        # Update step
        k = k + 1
        print(f"Actual step: {k}")

    # Exit
    return np.reshape(V@y, (256, 256))