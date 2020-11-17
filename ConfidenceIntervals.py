import numpy as np # https://numpy.org/doc/
#import statistics as stat # https://docs.python.org/3/library/statistics.html
import scipy as sp
from sklearn.utils import resample


def normal_CI(alpha, Scores, Hessian, theta_n_M):
    '''
    borders of a normal CI

    Arguments:
        - alpha: confidence parameter
        - Scores: matrix dim n x dim(theta) : Each rows is one S(theta|X_i)


        - Hessians: list of length n : Hessians at maximum estimate w.r.t. single Observations X_i | H(estimate|X_i)
                    - Every entry should be of the form: dim(theta_n_M) x dim(theta_n_M)

        - theta_n_M: The parameter estimate derived by multiple gradient ascent and comparison of likelihood values
                    - Should be of the form: 1x dim(theta_n_M)

    Outputs:
        - CI_borders: First row: Lower bound of CI // 1 x dim(theta_n_M)-vector
                      Second row:  Upper bound of CI // 1 x dim(theta_n_M)-vector

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
    H_n_inv = np.linalg.inv(Hessian)

    # Operations on Scores

    S_n = 1/n * np.dot(Scores.T, Scores) # 1/n sum S(theta|X_i) * S(theta|X_i)^T // is a dim(theta) by dim(theta) matrix

    # Cov
    Cov = np.dot(H_n_inv, np.dot(S_n, H_n_inv))
    Cov_diag = np.diag(Cov).reshape(1,-1)

    # Calculating bounds

    CI_borders = np.zeros((2, theta_n_M.shape[1]))

    CI_borders[0, :] = theta_n_M - z * Cov_diag
    CI_borders[1, :] = theta_n_M + z * Cov_diag

    print(f'Cov: {Cov}')
    print(f'z:{z}')

    return CI_borders




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
    n = X.shape[1]

    #Cache for saving Boostrap samples
    Bootstrap_thetas = mp.zeros((num_bootstraps, theta_hat.shape[1]))

    # Looping over amount of repitions of method
    for j in range(num_bootstraps):

        #Creating bootrap sample from original data
        X_bootstrap = resample(X, replace=True, n_samples= n)

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

    return CI_borders

