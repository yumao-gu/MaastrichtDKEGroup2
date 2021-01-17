import math
import datetime
import multiprocessing as mp
import sys
import torch
from torch.distributions import uniform, normal
from Auxillaries import *
from EM_ALG import *
from ConfidenceIntervals import *

# designing data structure to be used
weights = [.5,.45,.05]
means = [[0.],[.75],[2.75]]
covs = [[.2**2],[.2**2],[.2**2]]
theta_gt = np.array([[0.44246328, 1.0158333]])
get_data = lambda n: torch.from_numpy(gaussian_mixture_model_sample(n, means, covs, weights))

alpha = 0.1
type_CI = 'normal'
#type_CI = ''
# type_CI = 'bootstrap'
test_num = 10
n_power = 2
m = 5
n =math.pow(10,n_power)

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

    theta, _ = theta_n_M_EM(data = data,
                            n_runs = m,
                            max_iterations=2000)
  
    if type_CI == 'normal':
        # Getting Quantities that underly the CIs
        theta_hat = theta.clone().data.detach().numpy()
        Scores, Hessian = get_derivatives_torch(func=LogLikelihood,
                                                param=theta,
                                                data=data,
                                                print_info=False)
        ci, length = normal_CI(alpha, Scores, Hessian, theta_hat)
    elif type_CI == 'bootstrap':
        # Getting Quantities that underly the CIs
        theta = theta.clone().data.detach().numpy()
        ci, length, shape = boostrap_CI_torch_CG_FR(data, alpha, theta,
                                              num_bootstraps = 1000,
                                              func = LogLikelihood, lr = 0.01,
                                              n_iterations = 1000,print_info=False)
    else:
        print('Unknown type of CI!')
        sys.exit()

    end = time.time()
    #print(f'GetCI {end-start}')

    return ci, length


if __name__ == '__main__':
    GetCI(n,m, alpha, type_CI)