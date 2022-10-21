# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io

import os

import solvers, operators

# Import data
base_path = '.'

data = io.loadmat(os.path.join(base_path, './COULE_mat/280.mat'))
A = io.loadmat(os.path.join(base_path, 'A.mat'))['A']

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

# Problem parameters 
q = 1
rest = 30

# Compute solution
xTV = solvers.GraphLaTV(A, b, mu=150, R=3, sigmaInt=1e-3, q=1, n=n, m=m, rest=30)

# Save the solution
plt.imsave('xTV.png', xTV.reshape((m, n)), cmap='gray')