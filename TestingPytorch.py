import torch
from torch.distributions import uniform, normal
from Auxillaries import *
import math
import datetime
from ConfidenceIntervals import *

'''
Design of Model/Likelihood function to bet fitted
'''
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

    g_theta = (1-theta[0, 0])*phi_torch(x, 0, 0.2) + theta[0, 0]*phi_torch(x, theta[0, 1], 0.2)
    log_g_theta = torch.log(g_theta)

    return log_g_theta


'''
Create data of a gaussian mixture model
'''
# Specify parameters for mixture gaussian model
weights = [.5, .45, .05]
means = [[0.], [.75],[ 3.]]
covs = [[.2**2], [.2**2], [.2**2]]
n = 10000

# Create sampels of given model
X = gaussian_mixture_model_sample(n, means, covs, weights, test=False) # we can alternatively use pytorchs MixtureFamily
X = torch.from_numpy(X)


'''
Creating Contourlpot of likelihood function in dependence of parameters 
'''
# TODO: This is not well developed to be used for any setting: it needs limits for the parameters, works only for 2D, ... etc.
make_Contourplot = False
make_plots = True
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
# Set parameters here
n_iterations = 2500 #max number of iterations # TODO: alternatively/ additionally a error-based threshold?
lr = 0.005 # learning rate / stepsize
n_runs = 4

# Initializing Loss as minus infinity to make sure first run achieves higher likelihood
max_likelihood = -1*np.inf
# trajectory_dict is a cache to save the gradient ascent trajectory of all gradient ascent runs
trajectory_dict = {}

# Running Gradient Ascent multiple (M=n_runs) times
for run in range(n_runs):

    # Create/ Initialize variable ' TODO: make initialization more flexible
    theta = torch.tensor([[uniform.Uniform(0., .6).sample(),uniform.Uniform(0., 5.).sample()]], requires_grad = True)

    # Run complete Gradient ascent
    theta, L, trajectory = gradient_ascent_torch(func = LogLikelihood,
                                           param=theta,
                                           data=X,
                                           max_iterations=n_iterations,
                                           learningrate=lr,
                                           run_id=run,
                                           print_info=True)

    # Save optimization trajectory
    trajectory_dict.update({run : trajectory})
    # Updating Quantities if new max is found

    # compare likelihood value to previous runs
    if L > max_likelihood:
        # This takes forever if n is large. As it is torch implementation I don't see a way to get this faster
        print(f'New Maximum found! old:{max_likelihood} -> new:{L}')

        # Update highest likelihood and theta estimate
        max_likelihood = L
        theta_hat = theta.clone().data.numpy()

        # get derivatives
        Scores, Hessian = get_derivatives_torch(func=LogLikelihood,
                                                param=theta,
                                                data=X,
                                                print_dims=True)


CI = normal_CI(0.05, Scores, Hessian, theta_hat)

print(f'theta:\n {theta_hat}')
print(f'normal CI borders:\n {CI}')


CI = boostrap_CI_torch(data=X, alpha=0.05, theta_hat=theta_hat, num_bootstraps=10, func=LogLikelihood, lr=0.001,n_iterations = 200, print_ga_info=True)

print(f'theta:\n {theta_hat}')
print(f'Bootstrap CI borders:\n {CI}')

# Starting Plot
if make_plots:
    fig, ax = plt.subplots(1,1, figsize = (7,7))
    for run in range(n_runs):
        trajectory = trajectory_dict[run]
        m = len(trajectory_dict[run])
        ax.plot([trajectory[i][0,1] for i in range(m)], [trajectory[i][0,0] for i in range(m)], label = f'run: {run}')
    ax.imshow(Colourplot, cmap='Reds', extent = [0,5,0,.6], aspect='auto') #extent = [0,5,0,6]
    ax.set_xlabel('mu')
    ax.set_ylabel('rho')
    plt.title('Contourplot of Log-Likelihood function with gradient ascent trajectories')
    plt.show()


