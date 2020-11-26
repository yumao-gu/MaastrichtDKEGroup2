import numpy as np
import math
import datetime
import multiprocessing as mp

def GetCI(n,m):
    '''

    :param n: The number of splings
    :param m: The number of initializations
    :return: The confidence interval [low_bound,upper_bound]
    '''
    ci = [0, 1]
    return ci

def CISamplingTest(ground_truth,n_power,m,test_num):
    '''
    Analyze how the sampling number affects the CI.
    :param ground_truth: ground_truth value
    :param n_power: As for the probability convergence rate is sqrt(log(n)/n),
                    we design our experiment with n = 2^1,...2^i, with n the
                    number of sampling.
    :param m: The number of initalizations.
    :param test_num: The number of test iterations.
    :return:
    '''
    result = 0
    n = math.pow(2,n_power)
    for j in range(test_num):
        ci = GetCI(n, m)
        if ground_truth >= ci[0] and ground_truth <= ci[1]:
            result += 1
    return {n_power: result}


if __name__ == '__main__':
    start_t = datetime.datetime.now()
    num_cores = int(mp.cpu_count())
    print("the local computer has: " + str(num_cores) + " cpus")
    pool = mp.Pool(num_cores)
    params = []
    for i in range(21):1
        params.append([0.5,i,1,10000])
    results = [pool.apply_async(CISamplingTest, args=(ground_truth,n_power,m,test_num))
               for ground_truth,n_power,m,test_num in params]
    results = [p.get() for p in results]
    print(f'results {results}')
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("total cosuming time: " + "{:.2f}".format(elapsed_sec) + " 秒")