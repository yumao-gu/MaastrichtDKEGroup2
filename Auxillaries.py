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

    '''
    Arguemnts:
        - theta: a 1x2 matrix. First entry (theta[0,0]) is rho, the second (theta[0,1]) is mu
        - X: Matrix of observations: a 1xn matrix, where n is the number of samples

    Intermediate results:
        - g_theta: a 1xn matrix. The i-th entry is the density/likelihood of the fitted model w.r.t. X_i,
        - Likelihoods: a 1xn matrix this is \hat\L(\rho,\mu | X_i) - the log-likelihood w.r.t. X_i

    '''



    # Likelihood value
    g_theta = (1-theta[0,0])*norm.pdf(X,0,0.2) +theta[0,0]*norm.pdf(X,theta[0,1],0.2)
    Likelihoods = np.log(g_theta)
    L_value = np.mean(Likelihoods, axis = 1)

    # Derivatives first order // Score
    Score = np.zeros(shape = theta.shape)

    Scores_0 = np.exp(-1*Likelihoods)*(norm.pdf(X,theta[0, 1], 0.2)-norm.pdf(X, 0, 0.2))
    #print(f'Size Scores_0: {Scores_0.shape}')
    Score[0,0] = np.mean(Scores_0, axis = 1)

    Scores_1 = np.exp(-1*Likelihoods)*(norm.pdf(X,theta[0, 1], 0.2)*(X-theta[0,1])*(theta[0,0]/0.2**2))
    #print(f'Size Scores_1: {Scores_1.shape}')
    Score[0,1] = np.mean(Scores_1, axis = 1)
    


    return L_value, Score