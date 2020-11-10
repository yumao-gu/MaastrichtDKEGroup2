import numpy as np # https://numpy.org/doc/
#import statistics as stat # https://docs.python.org/3/library/statistics.html
import scipy as sp
import sympy as sym


def normal_CI(alpha, Scores, Hessians, theta_n_M):
    '''
    borders of a normal CI

    Arguments:
        - alpha: confidence parameter
        - Scores: list of length n : Scores at maximum estimate w.r.t. single Observations X_i | S(estimate|X_i)
                    - Every entry should be of the form: 1x dim(theta_n_M)

        - Hessians: list of length n : Hessians at maximum estimate w.r.t. single Observations X_i | H(estimate|X_i)
                    - Every entry should be of the form: dim(theta_n_M) x dim(theta_n_M)

        - theta_n_M: The parameter estimate derived by multiple gradient ascent and comparison of likelihood values
                    - Should be of the form: 1x dim(theta_n_M)

    Outputs:
        - lower: Lower bound of CI // 1 x dim(theta_n_M)-vector
        - upper: Upper bound of CI // 1 x dim(theta_n_M)-vector

    Further Information:
        - I expect the function tau to map the parameter vector to a single parameter,
          thus \nabla tau = e_i^T for some i = 1,..., dim(theta_n_M) .
          This results in diagonal entries of covariance matrix being considered
        - Confidence intervals are thus calculated for each component of the parameter theta_n_M

    TODO: Function testing: in particular dimensions
    '''

    assert len(Scores) == len(Hessians)
    assert 0 <= alpha and alpha <=1
    # assert dimensions of theta_n_M, Scores, Hessians

    # Calculate quantile
    z = sp.stats.norm.ppf(1-alpha/2)

    # Calculate Covariance matrix
    # Operations on Hessians
    H_n = np.mean(Hessians, axis=0) # This yields a dim(theta_n_M) x dim(theta_n_M) matrix
    H_n_inv = np.linalg.inv(H_n)

    # Operations on Scores
    S_rank_one_matrices = [np.dot(S.T,S) for S in Scores]
    S_n = np.mean(S_rank_one_matrices, axis=0)

    # Cov
    Cov = np.dot(H_n_inv, np.dot(S_n, H_n_inv))
    Cov_diag = np.diag(Cov).reshape(1,-1)

    # Calculating bounds
    lower = theta_n_M - z * Cov_diag
    upper = theta_n_M + z * Cov_diag

    return lower, upper




def boostrap_CI