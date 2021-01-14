import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal
from Auxillaries import *
import torch

def EMfromscratch(X,theta,it,con):
  x=X.numpy()
  mu,cov,rho=initial(theta)
  x=np.transpose(x)
  change=[1000,1000]
  for i in range(it):
    thetap=theta
    w=Estep(x,mu,cov,rho)
    mu,z=Mstep(x,w)
    theta=[z[1],mu[1]]

    thetat=torch.tensor([theta])
    loglikelihoods = LogLikelihood(thetat, X)
    loglikelihood_value = torch.mean(loglikelihoods)
    for i in range(len(theta)):
      change[i]=theta[i]-thetap[i]
      error=np.linalg.norm(change)
      if con>error:
        return thetat,loglikelihood_value
  return thetat,loglikelihood_value

def initial(theta):
  #mean initialization
  mu=[0.75,theta[1]]
  
  #cov based on paper
  cov=[0.2**2,0.2**2]

  #initialixetion of latent variables
  rho=theta[0]
  return mu,cov,rho
  
def Estep(x,mu,cov,rho):

  k=2
  pr=[]
  for i in range(k):
    pri=multivariate_normal.pdf(x,mu[i],cov[i])
    pr.append(pri)
  pr=np.array(pr)

  #comp of latent variables=>wi=(pdf(mui,covi)*zi)/sum(pdf(mui,covi)*zi)
  multofpr=[]
  z=[1-rho,rho]
  
  for i in range(k):
    multofpr.append(z[i]*pr[i])
  multofpr=np.array(multofpr)

  denom=np.zeros(x.shape[0])

  for j in range(x.shape[0]):
    for i in range(k):
      denom[j]=denom[j]+multofpr[i][j]

  w=np.zeros((k,x.shape[0]))
  
  for i in range(k):
    for j in range(x.shape[0]):
      w[i][j]=multofpr[i][j]/denom[j]
  return w

def Mstep(x,w):

  k=2
  d=w.shape[1]
  postw=np.zeros((1,k))

  for i in range(k):
    for j in range(x.shape[0]):
      postw[0][i]=postw[0][i]+w[i][j]
  zq=postw/x.shape[0]
  z=zq[0]

  
  mu=[0.75,0]
  wtr=w[1].reshape(len(x),1)
  numer=np.sum(wtr*x,axis=0)
  den=z[1]*d
  mutheta=numer/den
  mu[1]=mutheta[0]

  return mu,z


def theta_n_M_EM(data, n_runs, max_iterations=1000):

    # Initializing Loss as minus infinity to make sure first run achieves higher likelihood
    max_likelihood = -1 * np.inf



    for run in range(n_runs):
        theta = np.array([uniform.Uniform(0., .4).sample().numpy(),uniform.Uniform(0., 4.).sample().numpy()])

        # Run complete Gradient ascent
        theta, L = EMfromscratch(X=data,
                                  theta=theta,
                                  it=max_iterations,
                                  con=10**(-30))
        #print(f'{theta} {L}')

        # Updating Quantities if new max is found

        # compare likelihood value to previous runs
        if L > max_likelihood:
            # This takes forever if n is large. As it is torch implementation I don't see a way to get this faster
            # print(f'New Maximum found! old:{max_likelihood} -> new:{L}')

            # Update highest likelihood and theta estimate
            max_likelihood = L
            theta_hat = theta.clone().data.numpy()

    #print(f'theta_n_M_EM theta_hat {theta_hat}')
    theta_hat = torch.tensor(theta_hat, requires_grad = True)

    return theta_hat, max_likelihood


if __name__ == '__main__':
  weights = [.5,.45,.05]
  means = [[0.],[.75],[3.]]
  covs = [[.2**2],[.2**2],[.2**2]]
  theta_gt = np.array([[0.43109772,1.0575168]])
  get_data = lambda n: torch.from_numpy(gaussian_mixture_model_sample(n, means, covs, weights))
  data=get_data(int(10000))
  theta_n_M_EM(data, 100, max_iterations=100000)

