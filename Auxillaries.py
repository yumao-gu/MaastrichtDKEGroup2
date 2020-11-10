# Loading libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy as sp
from scipy.stats import norm

def gaussian_mixture_model(means, stds, weights, n_samples, dim=1):
    '''
    Create data of a gaussian mixture model

    Arguments:
        - means, stds: lists containing means and standard deviations of the single gaussian that are mixed
        - weights: weight of each used gaussian when combining
        - dim: Not sure if necessary: can be used for multidimensional gaussians
        - n_samples: the amount of data points to be sampled from the gaussian mixture model

    Outputs:
        - X: Data matrix containing the samples from the gaussian mixture model: (n_samples x dim)

    '''

    # make sure all lists are o same length and weights are well defined
    assert len(means) == len(stds)
    assert len(means) == len(weights)
    assert sum(weights) == 1

    # create cache for data
    X = np.zeros((dim,n_samples))

    # sample from each gaussian and add up to obtain mixture model
    for i in range(len(means)):
        X += weights[i] * np.random.normal(means[i], stds[i], (dim,n_samples))

    return X

def LL(theta, X):

    Score_list  = []
    Hessian_list = []
    L = 0
    for i in range(X.shape[1]):
        x = X[:,i]
        #print(x.shape)

        # Likelihood value
        f_theta = (1-theta[0,0])*norm.pdf(x,0,0.2) +theta[0,0]*norm.pdf(x,theta[0,1],0.2)
        L += np.log(f_theta)

        # Derivatives first order // Score
        Score = np.zeros(shape = (theta.shape[1], 1))

        d_weight_0 = norm.pdf(x,theta[0, 1], 0.2)-norm.pdf(x, 0, 0.2)
        Score[0,:] = d_weight_0

        d_weight_1 = theta[0, 0]*norm.pdf(x, theta[0, 1], 0.2)*(x-theta[0, 1])/(0.2**2)
        Score[1,:] = d_weight_1

        Score_list.append(Score)

        # Hessian
        # This does not work well yet for multiple datapoints at once
        Hessian = np.array([[0, norm.pdf(x, theta[0, 1], 0.2).squeeze()*(x.squeeze()-theta[0, 1])/(0.2**2)], [norm.pdf(x, theta[0, 1], 0.2).squeeze()*(x.squeeze()-theta[0, 1])/(0.2**2), norm.pdf(x, theta[0, 1], 0.2).squeeze()*((x.squeeze()-theta[0, 1])**2/(0.2**4)-1/(0.2**2))]])
        #print(x.shape, theta[0,0].shape,norm.pdf(x, theta[0, 1], 0.2).shape, (norm.pdf(x, theta[0, 1], 0.2)*((x-theta[0, 1])**2/(0.2**4)-1/(0.2**2))).shape)
        Hessian_list.append(Hessian)

    return L, Score_list, Hessian_list