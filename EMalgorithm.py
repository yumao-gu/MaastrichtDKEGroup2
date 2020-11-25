import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from scipy.stats import multivariate_normal
# use seaborn plotting defaults
# If this causes an error, you can comment it out.
#import seaborn as sns
#sns.set()


def EMsklearn(x,numofmodels):
  '''
  fit gaussian models using EM

  Arguments:
      - x: data 
      - numofmodels: number of gaussians to fit 

  Outputs:
      - mean
      -coveriance
      -density
  '''

  xn=np.zeros((len(x),1))
  for i in range(len(x)):
    xn[i]=x[i]

  clf= GaussianMixture(n_components=numofmodels, max_iter=500, random_state = 20).fit(xn)
  xpdf = np.linspace(-10,20,1000).reshape((-1,1))
  density = np.exp(clf.score_samples(xpdf))
  mean=clf.means_
  coveriance=clf.covariances_
  return density,mean,coveriance

def bicaic(xn):
  n_estimators = np.arange(1, 10)
  clfs = [GaussianMixture(n, max_iter=1000).fit(xn) for n in n_estimators]
  bics = [clf.bic(xn) for clf in clfs]
  aics = [clf.aic(xn) for clf in clfs]

  plt.plot(n_estimators, bics, label='BIC')
  plt.plot(n_estimators, aics, label='AIC')
  plt.legend();

def plotfitGauss(x,density):
  xpdf = np.linspace(-10,20,1000).reshape((-1,1))
  plt.hist(x, bins = 80, density = True, alpha=0.5)
  plt.plot(xpdf, density, '-r')
  plt.xlim(-10, 20);

def EMfromscratch(X,k,it):
  mu,cov,z=initial(X,k)
  x=np.transpose(X)
  
  for i in range(it):
    w=Estep(x,mu,cov,z)
    print('succsees estep')
    print('weights',w)
    print()
    mu,cov,z=Mstep(x,w)
    print('mu',mu)
    print()
    print('cov',cov)
    print()
    print('z',z)
    print('succsses mstep')
    
    print()

def initial(x,k):
  #mean initialization
  xdim=x.shape[0]
  mu=np.random.choice(x[1], (k,xdim))
  #print(mu)
  
  #semidefinetpositive initialized cov matrix
  cov=[]
  for i in range(k):
    cov.append(datasets.make_spd_matrix(xdim))
  cov=np.array(cov)
  #initialixetion of latent variables
  z=np.ones(k)
  for i in range(k):
    z[i]=1/k
  return mu,cov,z
  
def Estep(x,mu,cov,z):
  '''
  input data must be shape (datapoints,dimentions)
  '''
  k=len(z)
  pr=[]
  for i in range(k):
    #something might be wrong with the data shape ask*****
    pri=multivariate_normal.pdf(x,mu[i],cov[i])
    pr.append(pri)
  pr=np.array(pr)
  
  #comp of latent variables=>wi=(pdf(mui,covi)*zi)/sum(pdf(mui,covi)*zi)
  multofpr=[]
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

  k=w.shape[0]
  d=w.shape[1]
  xdim=x.shape[1]
  postw=np.zeros((1,k))
  for i in range(k):
    for j in range(x.shape[0]):
      postw[0][i]=postw[0][i]+w[i][j]
  zq=postw/x.shape[0]
  z=zq[0]

  mu=np.zeros((k,xdim))
  for i in range(k):
    wtr=w[i].reshape(len(x),1)
    numer=np.sum(wtr*x,axis=0)
    den=z[i]*d
    mu[i]=numer/den

  cov=np.zeros((k,xdim,xdim))
  for i in range(k):
    wtr=w[i].reshape(len(x),1)
    diff=x-mu[i]
    #print(diff)
    elm1=wtr*diff
    #print(elm1.shape)
    elm2=np.transpose(diff)
    numer=np.dot(elm2,elm1)
    #print(numer)
    den=z[i]*d
    cov[i]=numer/den

  return mu,cov,z
  
 








