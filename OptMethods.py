import torch
from torch.distributions import uniform, normal
from Auxillaries import *
import datetime

def NewtonMethod(func,data,lr,it,conv,print_info=False):
  theta = torch.tensor([[uniform.Uniform(0., .6).sample(),uniform.Uniform(0., 5.).sample()]], requires_grad = True)
  
  with torch.no_grad(): 
    optim_trajectory = [theta.clone().data.numpy()]
  err=10000
  for i in range(it):
    loglikelihoods = func(theta, data)

    loglikelihood_value = torch.mean(loglikelihoods)

    loglikelihood_value.backward()
    gradient = theta.grad

    #transpose for the multiplication
    grt=torch.transpose(gradient, 0, 1)

    # find inverse of hessian
    func_forHessian = lambda args: torch.mean(func(args, data))
    Hessian = torch.autograd.functional.hessian(func_forHessian, theta).squeeze()
    print(Hessian)
    HessianInverse=torch.inverse(Hessian)

    #search direction equivalent to the greadiand update in descent
    sd=torch.mm(HessianInverse, grt)
    #transpose back to 1*dim 
    sdt=torch.transpose(sd, 0, 1)

    with torch.no_grad():
            theta.add_(lr * sdt)
            theta.grad.zero_()
            optim_trajectory.append(theta.clone().data.numpy())

    if print_info:
      if i % 1 == 0:
        now = datetime.datetime.now()
        print(f' Iteration: {i} \t| Log-Likelihood:{loglikelihood_value} \t|  theta: {theta}  \t|Error:{err}  |  Time needed: {datetime.datetime.now()-now}  ')
    
    err=np.linalg.norm(optim_trajectory[-1] - optim_trajectory[-2])    
    if err < conv:
      break    

  return theta, loglikelihood_value, optim_trajectory




def ConjugateGradient_FletcherReeves(func,data,lr,it,conv,print_info=False):
  theta = torch.tensor([[uniform.Uniform(0., .4).sample(),uniform.Uniform(0., 4.).sample()]], requires_grad = True)
  
  with torch.no_grad(): 
    optim_trajectory = [theta.clone().data.numpy()]

  err=10000

  loglikelihoods = func(theta, data)
  loglikelihood_value = torch.mean(loglikelihoods)
  loglikelihood_value.backward()
  gradient = theta.grad
  gradientlist = [gradient.clone().data.numpy()]
  searchdirectionlist=[gradient.clone().data.numpy()]


  with torch.no_grad():
            theta.add_(lr * gradient)
            theta.grad.zero_()
            optim_trajectory.append(theta.clone().data.numpy())
  
  print(f' Iteration: {0} \t| Log-Likelihood:{loglikelihood_value} \t|  theta: {theta}  \t|Error:{err}')
  
  for i in range(it-1):
    loglikelihoods = func(theta, data)

    loglikelihood_value = torch.mean(loglikelihoods)

    loglikelihood_value.backward()
    gradient = theta.grad
    gradientlist.append(gradient.clone().data.numpy())


    #transpose for the multiplication
    grt=torch.transpose(gradient, 0, 1)

    #FletcherReeves scalar
    previousgradient=torch.tensor(gradientlist[i])
    previousgrt=torch.transpose(previousgradient, 0, 1)

    enm=torch.mm(gradient,grt)
    din=torch.mm(previousgradient,previousgrt)
    Beta=enm/din

    #previous search direction times FR scalar

    psd=torch.tensor(searchdirectionlist[i])
    addpart=psd*Beta

    #search direction equivalent to the greadiand update in descent
    sd=torch.add(gradient,addpart)
    searchdirectionlist.append(sd.clone().data.numpy())
    #print(sd)

    with torch.no_grad():
            theta.add_(lr * sd)
            theta.grad.zero_()
            optim_trajectory.append(theta.clone().data.numpy())

    if print_info:
      if i % 1 == 0:
        now = datetime.datetime.now()
        print(f' Iteration: {i+1} \t| Log-Likelihood:{loglikelihood_value} \t|  theta: {theta}  \t|Error:{err}  |  Time needed: {datetime.datetime.now()-now}  ')
    
    err=np.linalg.norm(optim_trajectory[-1] - optim_trajectory[-2])    
    if err < conv:
      break    

  return theta, loglikelihood_value, optim_trajectory





def ConjugateGradient_PolakRibiere(func,data,lr,it,conv,print_info=False):
  theta = torch.tensor([[uniform.Uniform(0., .6).sample(),uniform.Uniform(0., 5.).sample()]], requires_grad = True)
  
  with torch.no_grad(): 
    optim_trajectory = [theta.clone().data.numpy()]

  err=10000

  loglikelihoods = func(theta, data)
  loglikelihood_value = torch.mean(loglikelihoods)
  loglikelihood_value.backward()
  gradient = theta.grad
  gradientlist = [gradient.clone().data.numpy()]
  searchdirectionlist=[gradient.clone().data.numpy()]

  with torch.no_grad():
            theta.add_(lr * gradient)
            theta.grad.zero_()
            optim_trajectory.append(theta.clone().data.numpy())
  
  print(f' Iteration: {0} \t| Log-Likelihood:{loglikelihood_value} \t|  theta: {theta}  \t|Error:{err}')
  
  for i in range(it-1):
    loglikelihoods = func(theta, data)

    loglikelihood_value = torch.mean(loglikelihoods)

    loglikelihood_value.backward()
    gradient = theta.grad
    gradientlist.append(gradient.clone().data.numpy())

    #transpose for the multiplication
    grt=torch.transpose(gradient, 0, 1)

    #Polak-Ribiere scalar
    previousgradient=torch.tensor(gradientlist[i])
    previousgrt=torch.transpose(previousgradient, 0, 1)

    negpart=torch.add(gradient,-1*previousgradient)
    
    enm=torch.mm(negpart,grt)
    din=torch.mm(previousgradient,previousgrt)
    Beta=enm/din

    #previous search direction times pr scalar
    psd=torch.tensor(searchdirectionlist[i])
    addpart=psd*Beta

    #search direction equivalent to the greadiand update in descent
    sd=torch.add(gradient,addpart)
    searchdirectionlist.append(sd.clone().data.numpy())

    with torch.no_grad():
            theta.add_(lr * sd)
            theta.grad.zero_()
            optim_trajectory.append(theta.clone().data.numpy())

    if print_info:
      if i % 1 == 0:
        now = datetime.datetime.now()
        print(f' Iteration: {i+1} \t| Log-Likelihood:{loglikelihood_value} \t|  theta: {theta}  \t|Error:{err}  |  Time needed: {datetime.datetime.now()-now}  ')
    
    err=np.linalg.norm(optim_trajectory[-1] - optim_trajectory[-2])    
    if err < conv:
      break    

  return theta, loglikelihood_value, optim_trajectory