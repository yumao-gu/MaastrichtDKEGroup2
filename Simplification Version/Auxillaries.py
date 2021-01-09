# Loading libraries
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy as sp
import random
import torch
import datetime
from scipy.stats import norm,multivariate_normal

def gaussian_mixture_model_sample(n_samples = 10000,
                                  means = [[0.9, -0.8],[-0.7, 0.9]],
                                  covs = [[[2.0, 0.3], [0.3, 0.5]],
                                          [[0.3, 0.5], [0.3, 2.0]]],
                                  weights = [0.3,0.7],
                                  vis = False,
                                  print_info = False):

    '''
    Sample from a gaussian mixture model

    Arguments:
        - n_samples: the num of sample points
        - means, covs: lists containing means and standard deviations of \
                                        the single gaussian that are mixed. Note, the cov\
                                        matrix must be semidefine
        - weights: weight of each used gaussian when combining
        - test: test flag, only test dim = 2

    Outputs:
        - samples: Data matrix containing the samples from the gaussian mixture model
    '''
    # make sure all lists are o same length and weights are well defined
    start = time.time()
    np.random.seed()
    dim = len(means[0])
    samples = np.zeros((dim, n_samples))

    dim = len(means[0])
    samples = np.zeros((dim,n_samples))

    # sample from each gaussian and add up to obtain mixture model
    for i in range(n_samples):
        r = random.random()
        for j in range(len(weights)):
            if sum(weights[:j + 1]) > r:
                samples[:, i] = multivariate_normal.rvs(mean=means[j], cov=covs[j])
                break

    if vis and dim == 2:
        x = np.array(samples[0, :])
        y = np.array(samples[1, :])
        plt.scatter(x, y, alpha=0.2, s=1)
        plt.show()

    end = time.time()
    if print_info:
        print(f'gaussian_mixture_model_sample {end - start}')

    return samples

def phi_torch(x,mu,sigma):
    '''
    This function calculates the value of the density of a N(mu,sigma^2) gaussian random variable at point(s) x, only in a pytorch autograd compatible way

    Arguments:
        - x: a torch tensor. The output inherits the dimensions of x, as the density is applied elementwise
        - mu: a torch tensor / as a scalar: expected value of gaussian random variable
        - sigma: a torch tensor / as a scalar: standard deviation of gaussian random variable

    Output:
        - values of teh density at the provided x-values
        - as torch distributions deliver log_probs rather than probs we calculate prob = exp(log_prob)

    Old Calculation: - return  1/(torch.sqrt(torch.tensor([2*np.pi]))*sigma)*torch.exp(-(x-mu)**2/(2*sigma**2))
                     - don't necessarily need this function anymore, but it helps keeping things manageable

    '''

    # Initializing normal distribution
    distribution = torch.distributions.normal.Normal(mu, sigma)

    # Calculating log_probs
    log_prob = distribution.log_prob(x)

    # Calculating and returning probs via: prob = exp(log_prob)
    prob = torch.exp(log_prob)

    return prob

def LogLikelihood(theta, x):

    '''
    This is the (log-)likelihood function that is supposed to be fitted. Note that this is the (log-)likelihood w.r.t.
    one data point X_i. Thus, in later calculations like gradient ascent updates, the average of this function will be considered.
    This can easily achieved by torch.mean(LogLikelihood(theta, X)) where X is the complete data set (X_1, ..., X_n).

    Info:
        - phi_torch(x, mu, sigma) gives the value of a density of a N(mu,sigma)-random variable at point x

    Arguments:
        - theta: parameter vector that needs to altered to fit the model
        - x: a datapoint or a data set: the output will have the same dimensions as this

    Output:
        - model log-likelihood of theta given x // element-wise on x

    '''

    g_theta = (1-theta[0, 0])*phi_torch(x, 0, 0.2) + theta[0, 0]*phi_torch(x, theta[0, 1], 0.2)
    log_g_theta = torch.log(g_theta)

    return log_g_theta

def gradient_ascent_torch(func, param, data, max_iterations, learningrate, run_id = 0,  print_info = False):

    '''
    This function performs gradient ascent on the function func, which is governed by the arguments param.

    Arguments:
        - func: function to be maximized
        - param: torch tensor with gradient; parameters that serve as arguments of func
        - data: data that governs/parametrizes func. #TODO One might change the design to give the data/X to the function globally
        - max_iterations: int; (maximum) number of iterations to be performed during gradient ascent
        - learningrate: scalar; learning rate / step size of the algorithm
        - run_id: tracker of how many runs of the procedure have been done

    Outputs:
        - param: this (given convergence) is the argument of the maximum of func that was found.
        - loglikelihood_value: value of the found maximum
        - optim_trajectory: list of instances of param during optimization
    '''


    # starting time
    start = time.time()

    # save initial parameter to trajectory
    with torch.no_grad(): # necessary?
        optim_trajectory = [param.clone().data.numpy()]

    # Iterations
    for t in range(max_iterations):

        # Evaluate loglikelihood of each data point: L(param | X_i) for all i
        loglikelihoods = func(param, data) # has dimension 1 x num_data_points

        # Build mean of all log-likelihoods to get actual loglikelihood value
        loglikelihood_value = torch.mean(loglikelihoods) # has dim 1x1

        # Calculate gradients of param
        loglikelihood_value.backward()

        # Update param using gradient, save iterate and empty gradient for next calculation
        with torch.no_grad():
            param.add_(learningrate * param.grad)
            param.grad.zero_()
            optim_trajectory.append(param.clone().data.numpy())

        # Keeping informed of progress during optimization
    end = time.time()
    if print_info:
        print(f'gradient_ascent_torch {end-start}')

    # after all iterations are done return parameters, value of log-likelihood function at that maximum, trajectory
    return param, loglikelihood_value, optim_trajectory

def get_derivatives_torch(func, param, data, print_info = False):

    '''
    This function serves to calculate all the desired derivatives needed in the creation of CIs. This is based on torch.autograd.functional

    Arguments:
        - func: function of which the derivatives are to be calculated: this ought to be log(p(X_i | param)),
                that is function providing likelihood w.r.t. to each data point X_i
        - param: arguments of func, which are considered in the derivatives
        - data: data underlying teh log-likelihood function
        - print_dims: boolean whether to print dimensions of output or not // used for making suer dimensions are fitting

    Output:
        - Scores: n x dim(param) matrix. Scores[i,j] = S_j(param|X_i) = \nabla_{param_j}log(p(X_i | param))
        - Hessian: dim(param)x dim(param) matrix: Hessian[i,j] = \nabla_{param_j}\nabla_{param_i}  mean(log(p(X_s | param)), s=1,...,n)

    Procedure:
        - func, as being log(p(X_i | param)) cannot directly be used. giving the whole dataset X to func the element-wise application gives
          func(param, X) of size (dim(data)=1 x n_samples). Thus, fixing this as a function of param (c.f. 'func_forScore') we have that
          Scores = \nabla_{param} func(param, X) of (size n_samples x dim(param))
        - To calculate the hessian we need a scalar function we thus take the 'proper' log-likelihood function over the complete data set
          which is mean(log(p(X_s | param)) a function mapping from dim(param)->1.
          Thus, Hessian =  \nabla\nabla mean(log(p(X_s | param)), s=1,...,n) =  mean( \nabla\nabla log(p(X_s | param)), s=1,...,n),  as used in 'normal_CI'

    '''

    start = time.time()
    # Getting all scores w.r.t. the single X_i
    func_forScore = lambda args: func(args, data)
    Scores = torch.autograd.functional.jacobian(func_forScore, param).squeeze().squeeze()

    # hessian needs a scalar function
    func_forHessian = lambda args: torch.mean(func(args, data))
    Hessian = torch.autograd.functional.hessian(func_forHessian, param).squeeze().squeeze()

    end = time.time()
    if print_info:
        print(f'get_derivatives_torch {end-start}')

    return Scores, Hessian