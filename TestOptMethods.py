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

#NewtonMethod(LogLikelihood,X,0.001,100,10**(-5),True)

#ConjugateGradient_FletcherReeves(LogLikelihood,X,0.5,100,10**(-5),True)

ConjugateGradient_PolakRibiere(LogLikelihood,X,0.5,100,10**(-5),True)