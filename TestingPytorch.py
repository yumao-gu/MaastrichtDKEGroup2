import torch
from Auxillaries import *
import math
from hessian import hessian
import datetime


def phi_torch(x,mu,sigma):
    '''
    This function calculates the density of a N(mu,sigma^2) gaussian random variable in a pyotrch autograd compatible way

    Arguments:
        - x: a torch tensor. The output inherits the dimensions of x, as the density is applied elementwise
        - mu: a torch tensor / as a scalar: expected value of gaussian random variable
        - sigma: a torch tensor / as a scalar: standard deviation of gaussian random variable

    Output:
        - values of teh density at the provided x-values

    Old Calculation: return  1/(torch.sqrt(torch.tensor([2*np.pi]))*sigma)*torch.exp(-(x-mu)**2/(2*sigma**2))
    '''

    # Initilazing normal distribution
    distribution = torch.distributions.normal.Normal(mu, sigma)

    # Calculating logprobs
    log_prob = distribution.log_prob(x)

    # Calculating and returning probs
    return torch.exp(log_prob)

def LogLikelihood(theta, x):

    g_theta = (1-theta[0, 0])*phi_torch(x, 0, 0.2) + theta[0, 0]*phi_torch(x, theta[0, 1], 0.2)
    return torch.log(g_theta)


'''
Create data of a gaussian mixture model
'''
# Specify parameters for mixture gaussian model
weights = [.5, .45, .05]
means = [[0.], [.75],[ 3.]]
covs = [[.2**2], [.2**2], [.2**2]]
n = 10000

# Create sampels of given model
X = gaussian_mixture_model_sample(n, means, covs, weights, test=False)
X = torch.from_numpy(X)
print(f'Shape of X:{X.shape}')

'''
Creating Contourlpot of likelihood function in dependence of parameters 
'''
make_Contourplot = False
ticks = 1000

if make_Contourplot:
    print('Creating Colourplot...')
    x_lin = np.linspace(0, 5, ticks)  # range for mu
    y_lin = np.linspace(0, 0.6, ticks) # range for rho

    Colourplot = np.zeros((ticks, ticks))

    for i in reversed(range(ticks)):
        for j in range(ticks):
            x = x_lin[j]
            y = y_lin[ticks - i - 1]
            params = np.array([[y,x]])
            L, _ =LL(params, X)
            Colourplot[i, j] = L

    np.save('Contourmatrix', Colourplot)
else:
    print('Loading Colourplot...')
    Colourplot = np.load('Contourmatrix.npy')



'''
Gradient Ascent
'''
n_iterations = 3000
lr = 0.005
now = datetime.datetime.now()
n_runs = 4

# Initializing Loss as minusinifinty to make sure first run achieves higher likelihood
max_likelihood = -1*np.inf


# Running Gradient Ascent multiple times

trajectory_dict = {}

for run in range(n_runs):

    # Create/ Initialize variable
    theta = np.array([[np.random.uniform(0, .6), np.random.uniform(0, 5)]])
    theta = torch.from_numpy(theta)
    theta.requires_grad= True # to be randomly initialized

    with torch.no_grad():
        trajectory = [theta.data.numpy()]

    for t in range(n_iterations):

        loglikelihoods = LogLikelihood(theta, X)
        L = torch.mean(loglikelihoods)
        L.backward()

        with torch.no_grad():
            # Updating and clearing gradient
            theta.add_(lr * theta.grad)
            theta.grad.zero_()
            # Keeping track
            trajectory.append(theta.data.numpy())




        if t % 100 == 0:
            print(f'Run: {run+1}\t| Iteration: {t} \t| Log-Likelihood:{L} \t|  rho: {theta[0,0]}, mu: {theta[0,1]}  |  Time needed: {datetime.datetime.now()-now}  ')
            now = datetime.datetime.now()


    trajectory_dict.update({run : trajectory})
    # Updating Quantities if new max is found
    if L > max_likelihood:
        # Call LL again to obtian Hessian, too
        #loglikelihoods = LogLikelihood(theta, X)
        #L
        #L.backward()
        print('New Maximum found')
        # Update parameter, Scores and Hessian of currently best estimate
        #theta_hat = theta
        max_likelihood = L
        #Scores = S
        #Hessian = H

# Starting Plot
#fig, ax = plt.subplots(1,1, figsize = (7,7))
for run in range(n_runs):
    trajectory = trajectory_dict[run]
    plt.plot([a[0,1] for a in trajectory], [a[0,0] for a in trajectory], color = 'black')
#ax.imshow(Colourplot, cmap='Reds', extent = [0,5,0,.6], aspect='auto') #extent = [0,5,0,6]
#ax.set_xlabel('mu')
#ax.set_ylabel('rho')
#plt.title('Contourplot of Log-Likelihood function with gradient ascent trajectories')
plt.show()

