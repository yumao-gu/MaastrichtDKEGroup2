import numpy as np
import math,datetime,sys,torch,time,random,torch,datetime
import multiprocessing as mp
from torch.distributions import uniform, normal
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm,multivariate_normal
from sklearn.utils import resample

weights = [1]
means = [[0.]]
covs = [[1.]]
theta_gt = np.array([[0]])
get_data = lambda n: torch.from_numpy(gaussian_mixture_model_sample(n, means, covs, weights, vis=False))

def LogLikelihood(theta, x):

  distribution = torch.distributions.normal.Normal(theta[0, 0], 1)
  log_prob = distribution.log_prob(x)
  
  return log_prob

def gaussian_mixture_model_sample(n_samples = 10000,
                                  means = [[0.9, -0.8],[-0.7, 0.9]],
                                  covs = [[[2.0, 0.3], [0.3, 0.5]],
                                          [[0.3, 0.5], [0.3, 2.0]]],
                                  weights = [0.3,0.7],
                                  vis = False):

  start = time.time()
  np.random.seed()
  dim = len(means[0])
  samples = np.zeros((dim,n_samples))

  for i in range(n_samples):
    r = random.random()
    for j in range(len(weights)):
      if sum(weights[:j+1]) > r:
        samples[:,i] = multivariate_normal.rvs(mean = means[j], cov = covs[j])
        break

  end = time.time()
  print(f'gaussian_mixture_model_sample {end-start}')

  if vis and dim == 2:
    x = np.array(samples[0,:])
    y = np.array(samples[1,:])
    plt.scatter(x, y, alpha=0.2,s=1)
    plt.show()

  return samples

def gradient_ascent_torch(func, param , data, max_iterations, learningrate):

  start = time.time()
  for t in range(max_iterations):
    loglikelihoods = func(param, data)
    loglikelihood_value = torch.mean(loglikelihoods)
    # param.retain_grad()
    loglikelihood_value.backward()
    with torch.no_grad():
      param.add_(learningrate * param.grad)
      param.grad.zero_()
  
  end = time.time()
  print(f'gradient_ascent_torch {end-start}')

  return param, loglikelihood_value

def get_derivatives_torch(func, param, data):

  start = time.time()
  func_forScore = lambda args: func(args, data)
  Scores = torch.autograd.functional.jacobian(func_forScore, param).squeeze()

  func_forHessian = lambda args: torch.mean(func(args, data))
  Hessian = torch.autograd.functional.hessian(func_forHessian, param).squeeze()
  end = time.time()
  print(f'get_derivatives_torch {end-start}')

  return Scores, Hessian

def normal_CI(alpha, Scores, Hessian, theta_n_M):

  start = time.time()
  n = Scores.shape[0]
  z = sp.stats.norm.ppf(1-alpha/2)
  H_n_inv = np.linalg.inv(Hessian.reshape(theta_n_M.shape[1],theta_n_M.shape[1]))
  S_n = 1/n * np.dot(Scores.T, Scores)

  Cov = np.dot(H_n_inv, np.dot(S_n, H_n_inv))
  Cov_diag = np.diag(Cov).reshape(1,-1)
  Cov_diag = np.sqrt(1/n*Cov_diag)

  CI_borders = np.zeros((2, theta_n_M.shape[1]))
  CI_borders[0, :] = theta_n_M - z * Cov_diag ## CI_borders[0, i] = theta_n_M[i] - z*Cov_ii
  CI_borders[1, :] = theta_n_M + z * Cov_diag

  length = CI_borders[1, :] - CI_borders[0, :]
  shape = (CI_borders[1, :] - theta_n_M) / (theta_n_M - CI_borders[0, :])

  end = time.time()
  print(f'normal_CI {end-start}')

  return CI_borders, length, shape

def GetCI(n,m, alpha, type_CI):
    
  start = time.time()  
  data = get_data(int(n))
  theta = torch.tensor([[uniform.Uniform(-2, 2).sample()]], requires_grad=True)

  theta, _ = gradient_ascent_torch(func=LogLikelihood,
                                                param=theta,
                                                data=data,
                                                max_iterations=5000,
                                                learningrate=0.01)

  if type_CI == 'normal':
    theta_hat = theta.clone().data.detach().numpy()
    Scores, Hessian = get_derivatives_torch(func=LogLikelihood,
                                            param=theta,
                                            data=data)
    ci, length, shape = normal_CI(alpha, Scores, Hessian, theta_hat)
  
  end = time.time()
  print(f'GetCI {end-start}')

  return ci, length, shape

def CISamplingTest(ground_truth,n_power,m,test_num):

  start = time.time()
  result = 0
  length = 0
  shape = 0
  n = math.pow(10,n_power)

  num_cores = int(mp.cpu_count())
  print("the local computer has: " + str(num_cores) + " cpus")
  pool = mp.Pool(num_cores)
  params = []
  for i in range(test_num):
    params.append([n,m,alpha,type_CI])
  results = [pool.apply_async(GetCI, args=(n,m,alpha,type_CI))
               for n,m,alpha,type_CI in params]
  results = [p.get() for p in results]
  print(f'results {results}')

  for r in results:
    ci, lengthCI , shapeCI = r
    length += lengthCI.squeeze()
    shape += shapeCI.squeeze()
    if ground_truth >= ci[0] and ground_truth <= ci[1]:
      result += 1
  
  end = time.time()
  print(f'CISamplingTest {end-start}')

  return {n_power: (result/test_num, length/test_num, shape/test_num)}

alpha = 0.1
type_CI = 'normal'
test_num = 2
n_power = 5
m = 1

result = CISamplingTest(theta_gt[0,0],n_power,m,test_num)
print(f'result {result}')
