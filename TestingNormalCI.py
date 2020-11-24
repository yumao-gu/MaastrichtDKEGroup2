import torch
from torch.distributions import uniform, normal
from Auxillaries import *
import math
import datetime
from ConfidenceIntervals import *

'''
Design of Model/Likelihood function to bet fitted
'''
# For Testing: standard normal distribution with learnable parameters expected value and standard deviation
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

    g_theta = phi_torch(x, theta[0,0], theta[0,1])
    log_g_theta = torch.log(g_theta)

    return log_g_theta


'''
Create data of a gaussian mixture model
'''
# Specify parameters for mixture gaussian model
weights = [1]
means = [[0.]]
covs = [[1.]]
theta_gt = np.array([[0,1]])

type = 'normal'
n_CIs = 100
alpha = 0.05
theta_gt = np.array([[0,1]])
n_list = [10**i for i in range(1,7)]

n_ga_iterations =2500
n_ga_runs = 1
lr = 0.005
'''
Creating Contourlpot of likelihood function in dependence of parameters 
'''
# TODO: This is not well developed to be used for any setting: it needs limits for the parameters, works only for 2D, ... etc.
#make_Contourplot = False
#make_plots = False

CI_accuracy_dict = {}
for n in n_list:

    '''
    Creating n_CIs CIs. Storing borders in a dict.
    '''
    CI_border_dict = {}

    for CI_run in range(n_CIs):
        print('-'*10,f'CI run: {CI_run}','-'*10)
        # Create sampels of given model
        X = gaussian_mixture_model_sample(n, means, covs, weights, test=False) # we can alternatively use pytorchs MixtureFamily
        X = torch.from_numpy(X)


        '''
        Gradient Ascent
        '''
        # Set parameters here
        n_iterations = n_ga_iterations #max number of iterations # TODO: alternatively/ additionally a error-based threshold?
         # learning rate / stepsize
        n_runs = n_ga_runs

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
                                                   print_info=False)

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
                                                        print_dims=False)


        CI = normal_CI(alpha, Scores, Hessian, theta_hat)

        CI_border_dict.update({CI_run : CI})

    '''
    Testing CIs:
    '''
    print(f'All {n_CIs} normal CIs calculated! Testing these now!')
    # for mu = theta[0,0]
    sum_theta = np.zeros_like(theta_hat)

    for i in range(sum_theta.shape[1]):
        for CI_run in range(n_CIs):
            lower = CI_border_dict[CI_run][0,i]
            upper = CI_border_dict[CI_run][1,i]

            if lower <= theta_gt[0,i] <=upper:
                sum_theta[0,i]+=1

    sum_theta = sum_theta/n_CIs

    CI_accuracy_dict.update({n: sum_theta})

'''
Saving file
'''
saving_path = str(type)+'CI_accuracy_nCI'+str(n_CIs)+'_a'+str(alpha)
np.save(saving_path+'.npy', CI_accuracy_dict)

'''
Plot Accuracy in dependence of n
'''
CI_accuracies = np.array([CI_accuracy_dict[n] for n in  n_list])
print(CI_accuracies.shape)

for i in range(theta.shape[1]):
    plt.plot(n_list, CI_accuracies[:,0,i], label=f'theta_{i}')

plt.xlabel('number of samples')
plt.ylabel('CI coverage')
plt.hlines(1-alpha, xmin = np.min(n_list), xmax = np.max(n_list), linestyle ='dashed', linewidth = 0.4, color = 'grey', label = '1-alpha')
plt.legend()
plt.savefig(saving_path+'.png')
plt.show()


