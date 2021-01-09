import numpy as np
import math
import datetime
import multiprocessing as mp
import sys
import torch
from torch.distributions import uniform, normal
from Auxillaries import *
from ConfidenceIntervals import *

# designing data structure to be used
weights = [.5,.45,.05]
means = [[0.],[.75],[3.]]
covs = [[.2**2],[.2**2],[.2**2]]
theta_gt = np.array([[0.43309934,1.0573411]])
get_data = lambda n: torch.from_numpy(gaussian_mixture_model_sample(n, means, covs, weights))

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

#CI specifics

alpha = 0.1
type_CI = 'normal'
# type_CI = 'bootstrap'
test_num = 10
n_power = 5
m = 1

def GetCI(n,m, alpha, type_CI):
    '''
    :param n: The number of splings
    :param m: The number of initializations
    :param alpha: The accuracy parameter
    :param type_CI: string to determine what CI mthod to use / one of 'normal', 'bootstrap'
    :return: The confidence interval [low_bound,upper_bound]
    '''

    start = time.time()
    # Generate data
    data = get_data(int(n))

    #parameter to be optimized
    theta = torch.tensor([[uniform.Uniform(0.,.6).sample(),uniform.Uniform(0.,.5).sample()]], requires_grad=True)

    # gradient ascent
    theta, _, _ = gradient_ascent_torch(func=LogLikelihood,
                                                 param=theta,
                                                 data=data,
                                                 max_iterations=5000,
                                                 learningrate=0.1,
                                                 run_id=0,
                                                 print_info=False)

    if type_CI == 'normal':
        # Getting Quantities that underly the CIs
        theta_hat = theta.clone().data.detach().numpy()
        Scores, Hessian = get_derivatives_torch(func=LogLikelihood,
                                                param=theta,
                                                data=data,
                                                print_info=False)
        ci, length, shape = normal_CI(alpha, Scores, Hessian, theta_hat)
    elif type_CI == 'bootstrap':
        # Getting Quantities that underly the CIs
        theta = theta.clone().data.detach().numpy()
        ci, length, shape = boostrap_CI_torch(data, alpha, theta,
                                              num_bootstraps = 1000,
                                              func = LogLikelihood, lr = 0.01,
                                              n_iterations = 1000,print_info=False)
    else:
        print('Unknown type of CI!')
        sys.exit()

    end = time.time()
    print(f'GetCI {end-start}')

    return ci, length, shape

def CISamplingTest(ground_truth,n_power,m,test_num):
    '''
    Analyze how the sampling number affects the CI.
    :param ground_truth: ground_truth value
    :param n_power: As for the probability convergence rate is sqrt(log(n)/n),
                    we design our experiment with n = 2^1,...2^i, with n the
                    number of sampling.
    :param m: The number of initalizations.
    :param test_num: The number of test iterations.
    :return:
    '''

    start = time.time()
    result = 0
    length = 0
    shape = 0
    n = math.pow(10,n_power)

    num_cores = int(mp.cpu_count())
    print("the local computer has: " + str(num_cores) + " cpus")
    pool = mp.Pool(num_cores)
    params = []
    for i in range(test_num):
        params.append([n,m,alpha,type_CI])
    results = [pool.apply_async(GetCI, args=(n,m,alpha,type_CI))
               for n,m,alpha,type_CI in params]
    results = [p.get() for p in results]
    print(f'results {results}')

    for r in results:
        ci, lengthCI , shapeCI = r
        length += lengthCI.squeeze()
        shape += shapeCI.squeeze()
        if ground_truth[0][0] >= ci[0][0] and ground_truth[0][0] <= ci[1][0]\
                and ground_truth[0][1] >= ci[0][1] and ground_truth[0][1] <= ci[1][1]:
          result += 1

    end = time.time()
    print(f'CISamplingTest {end-start}')

    return {n_power: (result/test_num, length/test_num, shape/test_num)}

if __name__ == '__main__':
    result = CISamplingTest(theta_gt, n_power, m, test_num)
    print(f'result {result}')