from Auxillaries import *
from EMalgorithm import *
import numpy as np

weights = [.5, .5]
means = [[0.,0],[ 3.,5]]
covs = [[.2**2,], [.2**2]]
n = 4

# Create sampels of given model
X = gaussian_mixture_model_sample(50)
'''
to do:
check grafically
check n=4 for singular matrix
'''
print(f'Shape of X:{X.shape}')
'''
x=np.transpose(X)
#print(x)
mu,cov,z=initial(X,3)
#print(cov)
print(cov.shape)
w=Estep(x,mu,cov,z)
Mstep(x,w)
'''
EMfromscratch(X,2,10)