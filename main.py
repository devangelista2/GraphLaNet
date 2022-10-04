# Import libraries
from ast import operator
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io

import os
from numba import jit

import solvers, operators

# Import data
base_path = 'C:/Users/Tivogatto/OneDrive - Alma Mater Studiorum Università di Bologna/Desktop/GraphLaNet'

data = io.loadmat(os.path.join(base_path, './COULE_mat/0'))
A = io.loadmat(os.path.join(base_path, 'A.mat'))['A']

x_true = data['gt']
x_fbp = data['fbp']
b = A @ x_true.flatten()

# Add noise
delta = 6
b = b + delta * np.random.normal(0, 1, b.shape)

# Load informations
n = x_true.shape[0]
m = data['sino'].shape[0]

L = operators.GraphLaplacian(x_true, 1e-3, 10)
print(L.W.shape)