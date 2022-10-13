import scipy
from scipy import linalg
import numpy as np

v = np.array([[1], [3], [2]])
q, r = np.linalg.qr(v, mode='complete')

print(q)
print(r)