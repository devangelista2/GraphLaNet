# Import libraries
from matplotlib.pyplot import sci
import models 
import numpy as np

import tensorflow as tf
from tensorflow import keras as ks

from random import shuffle
import os

from skimage import transform

import scipy
from scipy import io

## ----------------------------------------------------------------------------------------------
## ---------- Initialization --------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------

# Load data
DATA_PATH = './COULE_mat/'

# Define the setup for the forward problem
noise_level = 0

# Setup parameters
start_angle = 0 # degrees
end_angle = 180 # degrees
n_angles = 120

theta = np.linspace(start_angle, end_angle, n_angles)

# Number of epochs
batch_size = 4
n_epochs = 50

## ----------------------------------------------------------------------------------------------
## ---------- Data Loader -----------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
class Data2D(ks.utils.Sequence):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size

        self.data_name_list = os.listdir(self.data_path)
        self.N = len(self.data_name_list)

        self.m, self.n = io.loadmat(os.path.join(self.data_path, self.data_name_list[0]))['gt'].shape

        # Shuffle data
        self.shuffled_idxs = np.arange(self.N)
        np.random.shuffle(self.shuffled_idxs)

    def __len__(self):
        'Number of batches per epoch'
        return int(self.N // self.batch_size)
    
    def __getitem__(self, idx):
        'Generate one batch of data'

        y = np.zeros((self.batch_size, self.m, self.n, 1))
        x = np.zeros((self.batch_size, self.m, self.n, 1))

        for i in range(self.batch_size):
            data = io.loadmat(os.path.join(self.data_path, self.data_name_list[i + self.batch_size*idx]))
            x_gt = data['gt']
            x_fbp = data['fbp']

            x[i, :, :, 0] = x_gt
            y[i, :, :, 0] = x_fbp.reshape((self.m, self.n))

        y = y.astype('float32')
        x = x.astype('float32')
            
        return y, x
## ----------------------------------------------------------------------------------------------
## ---------- NN --------------------------------------------------------------------------------
## ----------------------------------------------------------------------------------------------
# Define dataloader
trainloader = Data2D(DATA_PATH, batch_size=batch_size)

# Build model and compile it
model = models.get_UNet(input_shape = (256, 256, 1), n_scales = 4, conv_per_scale = 2, skip_connection=False)

# Define the Optimizer
initial_learning_rate = 5e-4
lr_schedule = ks.optimizers.schedules.PolynomialDecay(
    initial_learning_rate,
    decay_steps=1e4,
    end_learning_rate=1e-5)

model.compile(optimizer=ks.optimizers.Adam(learning_rate=initial_learning_rate),
              loss='mse')

# Train
model.fit(trainloader, epochs=n_epochs)
model.save(f"./model_weights/nn_unet_{noise_level}.h5")
print(f"Training of NN model -> Finished.")

# x_gt = io.loadmat(os.path.join(DATA_PATH, '280'))['gt']
# y_delta = transform.radon(x_gt, theta=theta, circle=False)
# y_delta = y_delta + noise_level * np.random.normal(0, 1, y_delta.shape)
# 
# x_fbp = transform.iradon(y_delta, theta=theta, circle=False)
# 
# print(A.shape)
# print(y_delta.shape)
# print(x_fbp.shape)
# 
# # Initialize A
# x = np.zeros((256, 256))
# x[0, 0] = 1
# 
# y = transform.radon(x_gt, theta=theta, circle=False)
# x_fbp = transform.iradon(y, theta=theta, circle=False)
# 
# import scipy.sparse
# A = scipy.sparse.csr_matrix(np.expand_dims(x_fbp.flatten(), 1))
# for i in range(256):
#     for j in range(256):
#         if i + j != 0:
#             x = np.zeros((256, 256))
#             x[i, j] = 1
# 
#             y = transform.radon(x_gt, theta=theta, circle=False)
#             x_fbp = transform.iradon(y, theta=theta, circle=False)
# 
#             A = scipy.sparse.hstack([A, np.expand_dims(x_fbp.flatten(), 1)])
#             print(A.shape)