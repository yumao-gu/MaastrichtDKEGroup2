from Auxillaries import *
from EMalgorithm import *

weights = [.5, .5]
means = [[0.],[ 3.]]
covs = [[.2**2], [.2**2]]
n = 10

# Create sampels of given model
X = gaussian_mixture_model_sample(n, means, covs, weights, test=False)
#print(X)
print(f'Shape of X:{X.shape}')
print(X[0])
x=X[0]
d,m,c=EMsklearn(x,2)
plotfitGauss(x,d)
