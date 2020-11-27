from Auxillaries import *
from EMalgorithm import *
import numpy as np

'''
to do:
check grafically
check 
'''
means = [[0.9, -0.8],[-0.7, 0.9]]
covs = [[[2.0, 0.3], [0.3, 0.5]],[[0.3, 0.5], [0.3, 2.0]]]
weights = [0.3,0.7]

X = gaussian_mixture_model_sample(500,means,covs,weights)
x=np.transpose(X)
print(f'Shape of X:{X.shape}')
m,c=EMsklearn(x,2,100)
print('estimated means and covariance using sklearn')
print(m)
print()
print(c)
ms,cs=EMfromscratch(X,2,100)
print('estimated means and covariance using emfromscratch')
print(ms)
print()
print(cs)