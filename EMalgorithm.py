import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture
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

