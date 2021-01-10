from OptMethods import *
import torch
from torch.distributions import uniform, normal
from Auxillaries import *
import math


weights = [.5, .45, .05]
means = [[0.], [.75],[ 3.]]
covs = [[.2**2], [.2**2], [.2**2]]
n = 1000

X = gaussian_mixture_model_sample(n, means, covs, weights, test=False) # we can alternatively use pytorchs MixtureFamily
X = torch.from_numpy(X)
#print(X)

def LogLikelihood(theta, x):

    g_theta = (1-theta[0, 0])*phi_torch(x, 0, 0.2) + theta[0, 0]*phi_torch(x, theta[0, 1], 0.2)
    log_g_theta = torch.log(g_theta)

    return log_g_theta

learningrate=0.001
max_itterations=10000
convergence_error=10**(-5)
print_info=True

'''
There is a possibility the hessian is not invertable if that is the case there is no solution
You have to re-run method suggestion we drop it or use Quassi-Newton

'''
#NewtonMethod(LogLikelihood,X,learningrate,max_itterations,convergence_error,print_info)

'''
ConjugateGradient_FletcherReeves
when lr is higher than 0.01 It can either converge really quick(less than 20 iteretions for error 10**(-5)) or take the wrong direction and blow the loglikelihood
so lr must be always low
'''
#ConjugateGradient_FletcherReeves(LogLikelihood,X,learningrate,max_itterations,convergence_error,print_info)

ConjugateGradient_PolakRibiere(LogLikelihood,X,learningrate,max_itterations,convergence_error,print_info)