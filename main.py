# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io

import os

import solvers, operators, utils

# Import data
base_path = '.'

data = io.loadmat('./COULE_mat/280.mat')
A = io.loadmat('./A.mat')['A']

x_true = data['gt']
x_fbp = data['fbp']
b = np.expand_dims(A @ x_true.flatten(), 1)

# Shave the shape of x
m, n = x_true.shape

# Flatten x
x_true = np.expand_dims(x_true.flatten(), 1)

print(f"Dimension of the problem: x -> {x_true.shape}, b -> {b.shape}")
print("")

# Add noise
delta = 0.5
b = b + delta * np.random.normal(0, 1, b.shape)

# Regularization parameter
mu = 10

# GraphLaplacian Parameters
sigmaInt = 1e-3
R = 3

"""
Starting point = x_true
"""
if True:
    # Get GraphLaplacian
    print("Creating Graph Laplacian")

    # Get the Graph Laplacian
    L = operators.GraphLaplacian(x_true.reshape((m, n)), sigmaInt, R).L # Line 9-11 of the pseudo-code
    print("Done!")
    print("")

    # Problem parameters 
    q = 1
    rest = 30

    # Compute solution
    xbest = solvers.GraphLaNet(A, b, L, mu=500, q=1, n=n, m=m, rest=30, show=True)

    # Save the solution
    plt.imsave('xbest.png', xbest.reshape((m, n)), cmap='gray')

"""
Starting point computed by some iteration of TV-Tikhonov regularization.
"""
if True:
    # Get GraphLaplacian
    print("Creating Graph Laplacian")

    # Compute the reference point x0 for the GraphLaplacian
    L = operators.TV(m, n)
    l = 20
    xGCV = utils.KTikhonovGenGCV(A, np.expand_dims(b.flatten(), 1), l, L)

    # Get the Graph Laplacian
    L = operators.GraphLaplacian(xGCV.reshape((m, n)), sigmaInt, R).L # Line 9-11 of the pseudo-code
    print("Done!")
    print("")

    # Problem parameters 
    q = 1
    rest = 30

    # Compute solution
    xTV = solvers.GraphLaNet(A, b, L, mu=150, q=1, n=n, m=m, rest=30, show=True)

    # Save the solution
    plt.imsave('xTV.png', xTV.reshape((m, n)), cmap='gray')

"""
Starting point computed by NN
"""
if False:
    # Get GraphLaplacian
    print("Creating Graph Laplacian")

    # Compute the reference point x0 for the GraphLaplacian
    weights_name = ""
    xNN = solvers.UNet(weights_name)

    # Get the Graph Laplacian
    L = operators.GraphLaplacian(xNN.reshape((m, n)), sigmaInt, R).L # Line 9-11 of the pseudo-code
    print("Done!")
    print("")

    # Problem parameters 
    q = 1
    rest = 30

    # Compute solution
    xLaNet = solvers.GraphLaNet(A, b, L, mu=150, q=1, n=n, m=m, rest=30, show=True)

    # Save the solution
    plt.imsave('xLaNet.png', xLaNet.reshape((m, n)), cmap='gray')