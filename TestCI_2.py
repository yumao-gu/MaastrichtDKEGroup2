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

    g_theta = phi_torch(x, theta[0, 0], 1)
    log_g_theta = torch.log(g_theta)

    return log_g_theta

# designing data structure to be used
weights = [1]
means = [[0.]]
covs = [[1.]]
theta_gt = np.array([[0]])
get_data = lambda n: torch.from_numpy(gaussian_mixture_model_sample(n, means, covs, weights, test=False))

#CI specifics
alpha = 0.1
type_CI = 'bootstrap'

def GetCI(n,m, alpha, type_CI):
    '''

    :param n: The number of splings
    :param m: The number of initializations
    :param alpha: The accuracy parameter
    :param type_CI: string to determine what CI mthod to use / one of 'normal', 'bootstrap'
    :return: The confidence interval [low_bound,upper_bound]
    '''
    # Generate data
    data = get_data(int(n))

    #parameter to be optimized
    theta = torch.tensor([[uniform.Uniform(-2, 2).sample()]], requires_grad=True)

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
        theta_hat = theta.clone().data.numpy()
        Scores, Hessian = get_derivatives_torch(func=LogLikelihood,
                                                param=theta,
                                                data=data,
                                                print_dims=False)
        ci = normal_CI(alpha, Scores, Hessian, theta_hat)
    elif type_CI == 'bootstrap':
        # Getting Quantities that underly the CIs
        theta = theta.clone().data.numpy()
        ci = boostrap_CI_torch(data, alpha, theta, num_bootstraps = 100, func = LogLikelihood, lr = 0.01, n_iterations = 100)
    else:
        print('Unknown type of CI!')
        sys.exit()

    return ci

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
    result = 0
    n = math.pow(2,n_power)
    for j in range(test_num):
        ci = GetCI(n,m, alpha=alpha, type_CI=type_CI)
        if ground_truth >= ci[0] and ground_truth <= ci[1]:
            result += 1
    return {n_power: result}


if __name__ == '__main__':
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("the local computer has: " + str(num_cores) + " cpus")
    pool = mp.Pool(num_cores)
    params = []
    for i in range(3,10):
        params.append([theta_gt[0,0],i,1,1000])
    results = [pool.apply_async(CISamplingTest, args=(ground_truth,n_power,m,test_num))
               for ground_truth,n_power,m,test_num in params]
    results = [p.get() for p in results]
    print(f'results {results}')
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("total cosuming time: " + "{:.2f}".format(elapsed_sec) + " 秒")