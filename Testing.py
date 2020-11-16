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
means = [[0.], [.75],[ 3.]]
covs = [[.2**2], [.2**2], [.2**2]]
n = 10000

# Create sampels of given model
X = gaussian_mixture_model_sample(n, means, covs, weights, test=False)
print(f'Shape of X:{X.shape}')


'''
Creating Contourlpot of likelihood function in dependence of parameters 
'''
make_Contourplot = False
ticks = 1000

if make_Contourplot:
    print('Creating Colourplot...')
    x_lin = np.linspace(0, 5, ticks)  # range for mu
    y_lin = np.linspace(0, 0.6, ticks) # range for rho

    Colourplot = np.zeros((ticks, ticks))

    for i in reversed(range(ticks)):
        for j in range(ticks):
            x = x_lin[j]
            y = y_lin[ticks - i - 1]
            params = np.array([[y,x]])
            L, _ =LL(params, X)
            Colourplot[i, j] = L

    np.save('Contourmatrix', Colourplot)
else:
    print('Loading Colourplot...')
    Colourplot = np.load('Contourmatrix.npy')


'''
Gradient Ascent
'''
n_iterations = 2000
lr = 0.005
now = datetime.datetime.now()
n_runs = 10

# Initializing Loss as minusinifinty to make sure first run achieves higher likelihood
max_likelihood = -1*np.inf

# Starting Plot
fig, ax = plt.subplots(1,1, figsize = (7,7))
# Running Gradient Ascent multiple times

for run in range(n_runs):

    theta = np.array([[np.random.uniform(0,.6), np.random.uniform(0,5)]])  # to be randomly initialized
    trajectory = [theta]

    for t in range(n_iterations):

        L, S, _ = LL(theta, X, calc_Hessian = False)
        theta = theta+lr*S
        trajectory.append(theta)

        if t % 100 == 0:
            print(f'Run: {run+1}\t| Iteration: {t} \t| Log-Likelihood:{L} \t|  rho: {theta[0,0]}, mu: {theta[0,1]}  |  Time needed: {datetime.datetime.now()-now}  ')
            now = datetime.datetime.now()

    ax.plot([theta[0,1] for theta in trajectory], [theta[0,0] for theta in trajectory], alpha = 0.8)

    # Updating Quantities if new max is found
    if L > max_likelihood:
        # Call LL again to obtian Hessian, too
        L, S, H = LL(theta, X, calc_Hessian = True)
        print('New Maximum found')
        # Update parameter, Scores and Hessian of currently best estimate
        theta_hat = theta
        max_likelihood = L
        Scores = S
        Hessians = H

ax.imshow(Colourplot, cmap='Reds', extent = [0,5,0,.6], aspect='auto') #extent = [0,5,0,6]
ax.set_xlabel('mu')
ax.set_ylabel('10*rho')
plt.title('Contourplot of Log-Likelihood function with gradient ascent trajectories')
plt.show()





