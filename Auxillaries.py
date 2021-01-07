# Loading libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random
import torch
import datetime
from scipy.stats import norm,multivariate_normal
test = True

def gaussian_model(pos = None, mean = [0.5, -0.2], cov = [[2.0, 0.3], [0.3, 0.5]], test = test):
    '''
    Get z value of a N-dim gaussian model

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

def gaussian_mixture_model(pos = None,means = [[0.9, -0.8],[-0.7, 0.9]],covs = [[[2.0, 0.3], [0.3, 0.5]],[[1.0, 0.8], [0.5, 0.9]]],weights = [0.3,0.7],test = test):

    '''
    Get z value of a gaussian mixture model

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

def gaussian_mixture_model_sample(n_samples = 10000,means = [[0.9, -0.8],[-0.7, 0.9]],covs = [[[2.0, 0.3], [0.3, 0.5]],[[0.3, 0.5], [0.3, 2.0]]],weights = [0.3,0.7],test = test):
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
    assert len(means) == len(covs)
    assert len(means) == len(weights)
    assert sum(weights) == 1

    dim = len(means[0])
    samples = np.zeros((dim,n_samples))

    # sample from each gaussian and add up to obtain mixture model
    for i in range(n_samples):
        r = random.random()
        for j in range(len(weights)):
            if sum(weights[:j+1]) > r:
                samples[:,i] = multivariate_normal.rvs(mean = means[j], cov = covs[j])
                break

    if test:
        assert dim == 2
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(samples[0,:],samples[1,:])
        plt.show()

    return samples

def LL(theta, X, calc_Hessian = False):

    '''
    Arguemnts:
        - theta: a 1x2 matrix. First entry (theta[0,0]) is rho, the second (theta[0,1]) is mu
        - X: Matrix of observations: a 1xn matrix, where n is the number of samples

    Intermediate results:
        - g_theta: a 1xn matrix. The i-th entry is the density/likelihood of the fitted model w.r.t. X_i,
        - Likelihoods: a 1xn matrix this is \hat\L(\rho,\mu | X_i) - the log-likelihood w.r.t. X_i
        - L_value: scalar; Actual likelihood value
        - E: a 1xn matrix; this gives 1/g_theta
        - phi1: a 1xn matrix; density values of one (mean mu) gaussian
        - phi2: a 1xn matrix; density values of second (mean zero) gaussian
        - Score: a nx2 matrix; Each row is one S(theta|X_i)

    '''

    n = X.shape[1]

    # Likelihood value
    g_theta = (1-theta[0,0])*norm.pdf(X,0,0.2) +theta[0,0]*norm.pdf(X,theta[0,1],0.2) # dim 1xn
    Likelihoods = np.log(g_theta) # dim 1xn
    L_value = np.mean(Likelihoods, axis = 1) # dim 1x1

    # Simplifications
    E = np.exp(-1*Likelihoods)      # dim 1xn
    phi1 = norm.pdf(X,theta[0, 1], 0.2)  # dim 1xn
    phi2 = norm.pdf(X,0, 0.2)  # dim 1xn

    # Derivatives first order // Score
    Score = np.zeros((n, theta.shape[1])) # dim nx2

    Score[:,0] = E*(phi1-phi2)   #dim 1xn
    Score[:,1] = E*(phi1*(X-theta[0,1])*(theta[0,0]/0.2**2)) #dim 1xn

    # Hessian
    H = np.zeros((theta.shape[1], theta.shape[1]))

    if calc_Hessian:
        # Deriavtive twice w.r.t rho
        d2_rho_summands = (phi1-phi2)**2 * E**2
        d2_rho = -np.mean(d2_rho_summands, axis = 1)
        H[0,0] = d2_rho

        # Deriavtive twice w.r.t mu
        d2_mu_summands = E*phi1*(((X-theta[0,1])/(0.2))**2-1) -theta[0,0]*E**2*phi1**2 * ((X-theta[0,1])/(0.2))**2
        d2_mu = theta[0,0]/0.2**2 * np.mean(d2_mu_summands , axis = 1)
        H[1,1] = d2_mu

        # Deriavtive wrt mu and rho
        d_mu_d_rho_summands = (X-theta[0,1])/0.2**2 * (E-theta[0,0]*E**2*(phi1-phi2))
        d_mu_d_rho = np.mean(d_mu_d_rho_summands, axis = 1)
        H[0,1]=H[1,0] = d_mu_d_rho



    return L_value, Score, H

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
    now = datetime.datetime.now()

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
        if print_info:
            if t % 100 == 0:
                # TODO make more flexible for any ind of parameter length
                print(f'Run: {run_id+1}\t| Iteration: {t} \t| Log-Likelihood:{loglikelihood_value} \t|  theta: {param}  |  Time needed: {datetime.datetime.now()-now}  ')
                now = datetime.datetime.now()

    # after all iterations are done return parameters, value of log-likelihood function at that maximum, trajectory
    return param, loglikelihood_value, optim_trajectory

def get_derivatives_torch(func, param, data, print_dims = False):

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
    # Getting all scores w.r.t. the single X_i
    func_forScore = lambda args: func(args, data)
    Scores = torch.autograd.functional.jacobian(func_forScore, param).squeeze().numpy()

    # hessian needs a scalar function
    func_forHessian = lambda args: torch.mean(func(args, data))
    Hessian = torch.autograd.functional.hessian(func_forHessian, param).squeeze().numpy()

    if print_dims:
        # sanity check
        print(f'Scores: {Scores}')
        print(f'Scores: {Scores.shape}')
        print(f'Actual Gradient: {np.mean(Scores, axis=0)} (~ 0)')
        print(f'Hessian {Hessian}')
        print(f'Hessian shape {Hessian.shape}')

    return Scores, Hessian

def load_data(filepath):

    '''
    This function is designed to load the data that was created using 'TestCI_2.py'. These is a list of dictionaries in a .txt file.
    Each of the dictionaries in the list has only one key, given by the sample size n the results were calculated on. Further,
    the value in each dictionary is a three tuple with the first entry being the coverage, the second the avg. length of CIS and
    the third the avg. shape of the CI.

    Arguments:
        - filepath: string with the filename/location were the data (as described above is stored)

    Output:
        - n_list, coverage, length, shape: lists of sample sizes used, list of corresponding coverage frequency


    '''
    import ast

    data = []
    with open(filepath, "r") as inFile:
        data = ast.literal_eval(inFile.read())

        n_list = []
    for dict_ in data:
        n_list.append(*dict_.keys())

    coverage = [data[i][n][0] for i, n in enumerate(n_list)]
    length = [data[i][n][1] for i, n in enumerate(n_list)]
    shape = [data[i][n][2] for i, n in enumerate(n_list)]

    return n_list, coverage, length, shape

def plot_data(n_list, coverage, length, shape, alpha):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    titles = ['Coverage proportion', 'Length of CIs', 'Shape of CIs']

    ax[0].plot(n_list, coverage)
    ax[0].hlines(1 - alpha, xmin=min(n_list), xmax=max(n_list), alpha=0.3, color='grey', linestyle='dashed',
                 label='1-α')
    ax[0].set_ylim((0,1))
    ax[1].plot(n_list, length)
    ax[2].plot(n_list, shape)

    for i in range(3):
        ax[i].set_xlabel('sample size: n')
        ax[i].set_title(titles[i])
        ax[i].set_xscale('log')

    ax[0].legend()
    plt.show()

def get_Vol(A,c):

    '''
    This funtions returns the volume of an elipse of the for  {y: y^T A y < c} for some symmetric positive definite A

    Arguments:
        -A: dxd matrix , symmetric, positive definite
        -c: non-negative scalar

    '''

    d,d1 = A.shape
    EigVals, _ = np.linalg.eig(A)

    assert  d == d1
    assert (A == A.T).all()
    assert (EigVals > 0).all()

    # Helpers
    det = np.prod(EigVals)**(-1/2)
    Gamma = sp.special.gamma(d/2+1)

    # Volume formula
    vol = det*(np.pi*c)**(d/2) / Gamma
    return vol































if __name__ == "__main__":
    # gaussian_model(test = test)
    # gaussian_mixture_model(test = test)
    gaussian_mixture_model_sample(test = test)
