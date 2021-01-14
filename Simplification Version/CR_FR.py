import numpy as np
import math
import datetime
import multiprocessing as mp
import sys
import torch
from torch.distributions import uniform, normal
from Auxillaries import *
from ConfidenceIntervals import *

# Loglikeihood fucntion to be fitted
# def LogLikelihood(theta, x):
#
#     '''
#     This is the (log-)likelihood function that is supposed to be fitted. Note that this is the (log-)likelihood w.r.t.
#     one data point X_i. Thus, in later calculations like gradient ascent updates, the average of this function will be considered.
#     This can easily achieved by torch.mean(LogLikelihood(theta, X)) where X is the complete data set (X_1, ..., X_n).
#
#     Info:
#         - phi_torch(x, mu, sigma) gives the value of a density of a N(mu,sigma)-random variable at point x
#
#     Arguments:
#         - theta: parameter vector that needs to altered to fit the model
#         - x: a datapoint or a data set: the output will have the same dimensions as this
#
#     Output:
#         - model log-likelihood of theta given x // element-wise on x
#
#     '''
#
#     g_theta = phi_torch(x, theta[0, 0], 1)
#     log_g_theta = torch.log(g_theta)
#
#     return log_g_theta

# designing data structure to be used

weights = [.5,.45,.05]
means = [[0.],[.75],[2.75]]
covs = [[.2**2],[.2**2],[.2**2]]
theta_gt = np.array([[0.44246328, 1.0158333]])
get_data = lambda n: torch.from_numpy(gaussian_mixture_model_sample(n, means, covs, weights))

#CI specifics
alpha = 0.1
type_CR = 'LogLike'
# type_CR = 'Score'
# type_CR = 'Wald'
test_num = 1000
n_power = 1
m = 1
n=1000
def GetCR(n,m, alpha, type_CR):
    '''

    :param n: The number of splings
    :param m: The number of initializations
    :param alpha: The accuracy parameter
    :param type_CR: string to determine what CI mthod to use / one of 'normal', 'bootstrap'
    :return: boolean whether gt is in CI or not
    '''
    # Generate data
    data = get_data(int(n))

    #parameter to be optimized
    theta = torch.tensor([[uniform.Uniform(0.,.6).sample(),uniform.Uniform(0.,.5).sample()]], requires_grad=True)

    theta, _, _ = theta_n_M_CG_FR(data=data,
                                  n_runs=m,
                                  func=LogLikelihood,
                                  max_iterations=2000,
                                  learningrate=0.01,
                                  print_info=False)

    if type_CR == 'LogLike':
        # Getting Quantities that underly the CIs
        theta_hat = theta.clone().data.numpy()
        '''Scores, Hessian = get_derivatives_torch(func=LogLikelihood,
                                                param=theta,
                                                data=data,
                                                print_dims=False)
        ci, length, shape = normal_CI(alpha, Scores, Hessian, theta_hat)'''
        ci_bool = LogLikeRatio_CR(data, alpha, theta_hat, theta_gt, func=LogLikelihood)

    elif type_CR == 'Score':
        # Getting Quantities that underly the CIs
        theta_hat = theta.clone().data.numpy()
        ci_bool = Score_CR(data, alpha, theta_gt, func = LogLikelihood )

    elif type_CR == 'Wald':
        # Getting Quantities that underly the CIs
        theta_hat = theta.clone().data.numpy()
        Scores, Hessian = get_derivatives_torch(func=LogLikelihood,
                                                param=theta,
                                                data=data,
                                                print_info=False)

        ci_bool = Wald_CR(data, alpha, theta_hat, theta_gt, Scores, Hessian)
    else:
        print('Unknown type of CI!')
        sys.exit()

    return ci_bool


if __name__ == '__main__':
    result = GetCR(n,m, alpha, type_CR)
    print(f'result {result}')