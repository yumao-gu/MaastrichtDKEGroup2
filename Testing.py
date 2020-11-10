# Loading libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy as sp
from scipy.stats import norm
from Auxillaries import *
import datetime

'''
Create data of a gaussian mixture model
'''
# Specify parameters for mixture gaussian model
weights = [.5, .45, .05]
means = [0., .75, 3]
stds = [.2, .2, .2]
n = 20000

# Create sampels of given model
X = gaussian_mixture_model(means, stds, weights, n_samples = n)
print(f'Shape of X:{X.shape}')
'''
Build Log-Likelihood function in dependence of parameters which are to be determined
'''
# initialize
theta = np.array([[0.4,2.6]]) #to be randomly initialized

# Gradient ascent
n_iterations = 100000
lr = 0.05
now = datetime.datetime.now()

for t in range(n_iterations):
    L, S, H = LL(theta, X)
    dW = np.mean(S,axis = 0)
    theta = theta + lr*dW.T

    if t % 1000 == 0:
        print(f'Iteration: {t} \t| Log-Likelihood:{L} \t| Time needed: {datetime.datetime.now()-now} | rho {theta[0,0]}, mu{theta[0,1]} ')
        now = datetime.datetime.now()




print(theta)
