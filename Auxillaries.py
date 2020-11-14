# Loading libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm,multivariate_normal
test = True

def gaussian_model(pos = None,
                                            mean = [0.5, -0.2],
                                            cov = [[2.0, 0.3], [0.3, 0.5]],
                                            test = test):
    '''
    Create data of a N-dim gaussian model

    Arguments:
        - means, cov: should be 1xN and NxN matrix
        - pos: the input of sample points
        - test: test flag

    Outputs:
        - value: : the gaussian output of sample points
    '''
    model = multivariate_normal(mean,cov)
    if test:
        x, y = np.mgrid[-1:1:.01, -1:1:.01]
        pos = np.dstack((x, y))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(x, y, model.pdf(pos))
        plt.show()
    return model.pdf(pos)

def gaussian_mixture_model(pos = None,
                                                                means = [[0.9, -0.8],[-0.7, 0.9]],
                                                                covs = [[[2.0, 0.3], [0.3, 0.5]],
                                                                                [[1.0, 0.8], [0.5, 0.9]]],
                                                                weights = [0.3,0.7],
                                                                test = test):
    '''
    Create data of a gaussian mixture model

    Arguments:
        - pos: the input of sample points
        - means, covs: lists containing means and standard deviations of \
                                        the single gaussian that are mixed. Note, the cov\
                                        matrix must be semidefine
        - weights: weight of each used gaussian when combining
        - test: test flag

    Outputs:
        - value: : the gaussian mixture output of sample points
    '''

    # make sure all lists are o same length and weights are well defined
    assert len(means) == len(covs)
    assert len(means) == len(weights)
    assert sum(weights) == 1
    if not test:
        assert len(pos) > 0

    if test:
        x, y = np.mgrid[-2:2:.01, -2:2:.01]
        pos = np.dstack((x, y))

    value = np.zeros(pos[:,:,0].shape)
    # sample from each gaussian and add up to obtain mixture model
    for i in range(len(means)):
        value += weights[i] * gaussian_model(pos,means[i], covs[i],False)

    if test:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.contourf(x, y, value,levels = 30)
        plt.show()

    return value

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

if __name__ == "__main__":
    # gaussian_model(test = test)
    gaussian_mixture_model(test = test)
