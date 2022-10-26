# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io

import os

import solvers, operators, utils, metrics

# Import data
base_path = '.'

data = io.loadmat('./COULE_mat/280')
A = io.loadmat('./A.mat')['A']

x_true = data['gt']
x_fbp = data['fbp']
b = np.expand_dims(A @ x_true.flatten(), 1)

# Save x_true for visualization
plt.imsave('results/xtrue.png', x_true, cmap='gray')
plt.imsave('results/xFBP.png', x_fbp, cmap='gray')

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
Starting point = x_fbp
"""
if True:
    # Get GraphLaplacian
    print("Creating Graph Laplacian")

    # Get the Graph Laplacian
    L = operators.GraphLaplacian(x_fbp.reshape((m, n)), sigmaInt, R).L # Line 9-11 of the pseudo-code
    print("Done!")
    print("")

    # Problem parameters 
    q = 1
    rest = 30

    # Compute solution
    xLaFBP = solvers.GraphLaNet(A, b, L, mu=5000, q=1, n=n, m=m, rest=30, show=True)

    # Save the solution
    plt.imsave('results/xLaFBP.png', xLaFBP.reshape((m, n)), cmap='gray')

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
    xLaTrue = solvers.GraphLaNet(A, b, L, mu=5000, q=1, n=n, m=m, rest=30, show=True)

    # Save the solution
    plt.imsave('results/xLaTrue.png', xLaTrue.reshape((m, n)), cmap='gray')

"""
Starting point computed by some iteration of TV-Tikhonov regularization.
"""
if True:
    # Get GraphLaplacian
    print("Creating Graph Laplacian")

    # Compute the reference point x0 for the GraphLaplacian
    L = operators.TV(m, n)
    l = 20
    xTV = solvers.TikTV(A, np.expand_dims(b.flatten(), 1), l, L)

    # Get the Graph Laplacian
    L = operators.GraphLaplacian(xTV.reshape((m, n)), sigmaInt, R).L # Line 9-11 of the pseudo-code
    print("Done!")
    print("")

    # Problem parameters 
    q = 1
    rest = 30

    # Compute solution
    xLaTV = solvers.GraphLaNet(A, b, L, mu=5000, q=1, n=n, m=m, rest=30, show=True)

    # Save the solution
    plt.imsave('results/xTV.png', xTV.reshape((m, n)), cmap='gray')
    plt.imsave('results/xLaTV.png', xLaTV.reshape((m, n)), cmap='gray')

"""
Starting point computed by NN
"""
if True:
    # Get GraphLaplacian
    print("Creating Graph Laplacian")

    # Compute the reference point x0 for the GraphLaplacian
    weights_name = "nn_unet_0"
    #model = solvers.UNet(weights_name)
    #xNet = utils.predict(model, x_fbp)
    xNet = data['recon']

    # Get the Graph Laplacian
    L = operators.GraphLaplacian(xNet.reshape((m, n)), sigmaInt, R).L # Line 9-11 of the pseudo-code
    print("Done!")
    print("")

    # Problem parameters 
    q = 1
    rest = 30

    # Compute solution
    xLaNet = solvers.GraphLaNet(A, b, L, mu=5000, q=1, n=n, m=m, rest=30, show=True)

    # Save the solution
    plt.imsave('results/xNet.png', xNet.reshape((m, n)), cmap='gray')
    plt.imsave('results/xLaNet.png', xLaNet.reshape((m, n)), cmap='gray')


"""
Compute the metrics
"""
if True:
    # Relative error
    print(f"RE(x_true, xFBP) = {metrics.rel_err(x_fbp, x_true)}")
    print(f"RE(x_true, xLaFBP) = {metrics.rel_err(xLaFBP, x_true)}")
    print(f"RE(x_true, xLaTrue) = {metrics.rel_err(xLaTrue, x_true)}")
    print(f"RE(x_true, xTV) = {metrics.rel_err(xTV, x_true)}")
    print(f"RE(x_true, xLaTV) = {metrics.rel_err(xLaTV, x_true)}")
    print(f"RE(x_true, xNet) = {metrics.rel_err(xNet, x_true)}")
    print(f"RE(x_true, xLaNet) = {metrics.rel_err(xLaNet, x_true)}")
    print("")

    # PSNR
    print(f"PSNR(x_true, xFBP) = {metrics.PSNR(x_true, x_fbp)}")
    print(f"PSNR(x_true, xLaFBP) = {metrics.PSNR(x_true, xLaFBP)}")
    print(f"PSNR(x_true, xLaTrue) = {metrics.PSNR(x_true, xLaTrue)}")
    print(f"PSNR(x_true, xTV) = {metrics.PSNR(x_true, xTV)}")
    print(f"PSNR(x_true, xLaTV) = {metrics.PSNR(x_true, xLaTV)}")
    print(f"PSNR(x_true, xNet) = {metrics.PSNR(x_true, xNet)}")
    print(f"PSNR(x_true, xLaNet) = {metrics.PSNR(x_true, xLaNet)}")