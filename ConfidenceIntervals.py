import numpy as np # https://numpy.org/doc/
#import statistics as stat # https://docs.python.org/3/library/statistics.html
import scipy as sp
from sklearn.utils import resample
from Auxillaries import *


def normal_CI(alpha, Scores, Hessian, theta_n_M):
    '''
    This function calculates the borders of a normal CI

    Arguments:
        - alpha: confidence parameter
        - Scores: matrix dim n x dim(theta) : Each rows is one S(theta|X_i) (transposed)

        - Hessian:  dim(theta_n_M) x dim(theta_n_M) = 1/n sum H(theta | X_i)


        - theta_n_M: The parameter estimate derived by multiple gradient ascent and comparison of likelihood values
                    - Should be of the form: 1x dim(theta)

    Outputs:
        - CI_borders: First row: Lower bounds of CI // 1 x dim(theta_n_M)-vector
                      Second row:  Upper bounds of CI // 1 x dim(theta_n_M)-vector

    Further Information:
        - I expect the function tau to map the parameter vector to a single parameter,
          thus \nabla tau = e_i^T for some i = 1,..., dim(theta_n_M) .
          This results in diagonal entries of covariance matrix being considered
        - Confidence intervals are thus calculated for each component of the parameter theta_n_M

    TODO: Function testing: in particular dimensions
    '''

    assert 0 <= alpha and alpha <=1
    # assert dimensions of theta_n_M, Scores, Hessians

    n = Scores.shape[0]

    # Calculate quantile
    z = sp.stats.norm.ppf(1-alpha/2)

    # Calculate Covariance matrix
    # Operations on Hessians
    H_n_inv = np.linalg.inv(Hessian.reshape(theta_n_M.shape[1],theta_n_M.shape[1]))

    # Operations on Scores
    S_n = 1/n * np.dot(Scores.T, Scores) # 1/n sum S(theta|X_i) * S(theta|X_i)^T // is a dim(theta) by dim(theta) matrix

    # Cov
    Cov = np.dot(H_n_inv, np.dot(S_n, H_n_inv))
    Cov_diag = np.diag(Cov).reshape(1,-1)
    Cov_diag = np.sqrt(1/n*Cov_diag) # I think thats missing in the formlua

    # Calculating bounds

    CI_borders = np.zeros((2, theta_n_M.shape[1]))

    CI_borders[0, :] = theta_n_M - z * Cov_diag ## CI_borders[0, i] = theta_n_M[i] - z*Cov_ii
    CI_borders[1, :] = theta_n_M + z * Cov_diag

    # Getting further quantities
    length = CI_borders[1, :] - CI_borders[0, :]
    shape = (CI_borders[1, :] - theta_n_M) / (theta_n_M - CI_borders[0, :])

    return CI_borders, length, shape




def boostrap_CI(X, alpha, theta_hat, num_bootstraps, lr, n_iterations = 100 ):

    '''
    Function to create CI via bootstrap method

    Arguments:
        - X: original dataset
        - alpha: confidence parameter
        - theta_hat: Estimate derived by the normal M-times run gradient ascent
        - num_bootstraps: number of bootstrap samples to be created
        - lr: learning rate used in gradient ascent
        - n_iterations: number of steps used in gradient ascent

    Intermediates:
        - n: size of provided dataset
        - Bootstrap_thetas: Cache to store the new, to be calculated, estimates of theta, dim: num_bootstraps x dim theta
        - X_bootstrap: Bootstrap sample of X, size n
        - CI_borders: lower and upper bounds of CI of each theta_i : dim 2x dim theta

    Further Information:
        - I expect the function tau to map the parameter vector to a single parameter,
          thus we create a CI for every entry of the parameter vector.

    '''

    # number of data points
    dimX, n = X.shape

    #Cache for saving Boostrap samples
    Bootstrap_thetas = np.zeros((num_bootstraps, theta_hat.shape[1]))

    # Looping over amount of repitions of method
    for j in range(num_bootstraps):

        #Creating bootrap sample from original data
        X_bootstrap = resample(X.squeeze(), replace=True, n_samples= n).reshape(X.shape)

        # perform GA on new likelihhod function
        theta = theta_hat # initialize theta as theta_hat

        for t in range(n_iterations):
            L, S, _ = LL(theta, X_bootstrap, calc_Hessian=False)
            theta = theta + lr * S

            #TODO This could do with some checking whether it converged or not. this also should need very little iterations -> really small lr?

        # Saving sample of theta
        Bootstrap_thetas[j,:] = theta

    # Cache for borders of CIs first row lower bound, second row upper bound, each column corresponds to one entry of the parameter vector
    CI_borders = np.zeros((2,theta_hat.shape[1]))

    CI_borders[0,:] = np.quantile(Bootstrap_thetas, alpha/2, axis = 0)
    CI_borders[1,:] = np.quantile(Bootstrap_thetas, 1-alpha/2, axis = 0)

    print(f'CI: {CI}')

    return CI_borders


def boostrap_CI_torch(data, alpha, theta_hat, num_bootstraps, func, lr, n_iterations = 200, print_ga_info = False ):

    '''
    Function to create CI via bootstrap method

    Arguments:
        - X: original dataset
        - alpha: confidence parameter
        - theta_hat: Estimate derived by the normal M-times run gradient ascent
        - num_bootstraps: number of bootstrap samples to be created
        - lr: learning rate used in gradient ascent
        - n_iterations: number of steps used in gradient ascent

    Intermediates:
        - n: size of provided dataset
        - Bootstrap_thetas: Cache to store the new, to be calculated, estimates of theta, dim: num_bootstraps x dim theta
        - X_bootstrap: Bootstrap sample of X, size n
        - CI_borders: lower and upper bounds of CI of each theta_i : dim 2x dim theta

    Further Information:
        - I expect the function tau to map the parameter vector to a single parameter,
          thus we create a CI for every entry of the parameter vector.

    '''

    # number of data points
    dim, n = data.shape

    #Cache for saving Boostrap samples
    Bootstrap_thetas = np.zeros((num_bootstraps, theta_hat.shape[1]))

    # Looping over amount of repitions of method
    for j in range(num_bootstraps):

        #Creating bootrap sample from original data
        X_bootstrap = data[:, np.random.choice(n, size=n, replace=True)].reshape(data.shape)

        # perform GA on new likelihhod function
        theta = torch.tensor(theta_hat, requires_grad = True) # initialize theta as theta_hat

        theta, _, _ = gradient_ascent_torch(func=func,
                                            param=theta,
                                            data=X_bootstrap,
                                            max_iterations=n_iterations,
                                            learningrate=lr,
                                            print_info=print_ga_info)

            #TODO This could do with some checking whether it converged or not. this also should need very little iterations -> really small lr?

        # Saving sample of theta
        Bootstrap_thetas[j,:] = theta.clone().data.numpy() # size n x dim(theta)

    # Cache for borders of CIs first row lower bound, second row upper bound, each column corresponds to one entry of the parameter vector
    CI_borders = np.zeros((2,theta_hat.shape[1]))

    CI_borders[0,:] = np.quantile(Bootstrap_thetas, alpha/2, axis = 0)
    CI_borders[1,:] = np.quantile(Bootstrap_thetas, 1-alpha/2, axis = 0)

    # Getting further quantities
    length = CI_borders[1,:]-CI_borders[0,:]
    shape = (CI_borders[1,:]-theta_hat)/(theta_hat-CI_borders[0,:])

    return CI_borders, length, shape

