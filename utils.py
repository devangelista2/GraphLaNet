import numpy as np
from scipy import optimize, sparse, linalg

def updateQR(A, L, AV, LV, QA, RA, QL, RL, V, r):
    # Updates all the matrices in the new (k+1)-th dimensional Krylov space.

    # Update for A
    vn = r / np.linalg.norm(r) # Line 24
    Avn = A @ vn
    AV = np.concatenate([AV, Avn], axis=1) # AV_k+1 = [AV_k, vnew]
    
    # Update for V
    Lvn = L @ vn
    V = np.concatenate([V, vn], axis=1)   # LV_k+1 = [LV_k, vnew]
    LV = np.concatenate([LV, Lvn], axis=1)

    # Update for QA and RA such that A = QA * RA
    rA = QA.T @ Avn
    qA = Avn - QA @ rA
    ta = np.linalg.norm(qA)
    qtA = qA / ta
    QA = np.concatenate([QA, qtA], axis=1)
    RA = np.concatenate([np.concatenate([RA, rA], axis=1),
                         np.concatenate([np.zeros((1, len(rA))), np.array([[ta]])], axis=1)], axis=0)
    
    # Update for QL and RL such that A = QA * RA
    rL = QL.T @ Lvn
    qL = Lvn - QL @ rL
    tL = np.linalg.norm(qL)
    qtL = qL / tL
    QL = np.concatenate([QL, qtL], axis=1)
    RL = np.concatenate([np.concatenate([RL, rL], axis=1), 
                         np.concatenate([np.zeros((1, len(rL))), np.array([[tL]])], axis=1)], axis=0)

    return AV, LV, QA, RA, QL, RL, V

def KTikhonovGenGCV(A, b, k, L, mu=145):
    # Solves the Tikhonov problem in general form 
    # x = argmin || Ax - b ||^2 + mu * ||Lx||
    # in the GK Krylov subspace of dimension k, with mu set by the user.

    # Compute the matrices B and V of Lanczos bidiagonalization
    _, B, V = lanc_b(A, b, k)

    # Define vector e with e[0] = || b ||, e[1:] = 0
    e = np.zeros((2*k+1, 1))
    e[0] = np.linalg.norm(b)

    # Compute the QR of LV
    LV = L @ V
    _, R = np.linalg.qr(LV, 'reduced')
    
    # Compute the solution y in the GK Krylov subspace
    y = np.linalg.lstsq(np.concatenate([B, np.sqrt(mu) * R], axis=0), e, rcond=-1)[0]
    return V @ y


def lanc_b(A, p, k):
    # Perform Lanczos bidiagonalization with/without reorthogonalization
    m, n = A.shape

    # Initialize the matrices U, V, B with the right shape
    U = np.zeros((m, k+1))
    V = np.zeros((n, k))
    B = np.zeros((k+1, k))

    # Prepare for Lanczos iteration
    v = np.zeros((n, 1))
    beta = np.linalg.norm(p)

    if beta == 0:
        raise ValueError
    
    # Initialize matrix U
    u = p / beta
    U[:, 0] = u.flatten()

    # Build U, B, V element by element
    for i in range(k):

        # Compute the residual r
        r = A.T @ u - beta * v
        for j in range(i-1):
            r = r - (V[:, j].T @ r) * np.expand_dims(V[:, j], 1)
        
        # Diagonal of B and columns of V
        alpha = np.linalg.norm(r)
        v = r / alpha 
        B[i, i] = alpha
        V[:, i] = v.flatten()

        # Compute the residual p
        p = A @ v - alpha * u
        for j in range(i):
            p = p - (U[:, j].T @ p) * np.expand_dims(U[:, j], 1)

        # Over-diagonal of B and columns of U
        beta = np.linalg.norm(p)
        u = p / beta
        B[i+1, i] = beta
        U[:, i+1] = u.flatten()
        

    return U, B, V