import torch
import sys
sys.path.append('../src')
sys.path.append('../src/bern_cuda_ext')
from poly_utils import *
from relu_coeffs import *
from network_modules_list import *
import ibf_minmax_cpp
import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(0)

    n_vars_list = np.arange(2, 6)
    n_trials = 5
    runtimes = np.empty((n_trials, n_vars_list.shape[0]))
    for idx, n_vars in enumerate(n_vars_list):
        for trial in range(3): 
            get_time(n_vars)
        for trial in range(n_trials): 
            dur = get_time(n_vars)
            runtimes[trial, idx] = dur
    print(runtimes)
    plt.boxplot(runtimes, labels=n_vars_list)
    plt.savefig('dim_vs_time.pdf')

def get_time(n_vars):
    start = time.time()
    exp(n_vars)
    torch.cuda.synchronize()
    end = time.time()
    return end - start

def exp(n_vars):
    intervals = [[-1, 1] for i in range(n_vars)]
    intervals = torch.tensor(intervals, dtype = torch.float32)
    inputs = generate_inputs(n_vars, intervals, device = 'cuda')
    # print(inputs)
    network_size =  [n_vars, 20, 20, 1]
    network_weights = []
    network_biases = []

    for i in range(len(network_size) - 1):
        weights = torch.randn(network_size[i + 1], network_size[i])
        biases = torch.randn(network_size[i + 1], 1)
        network_weights.append(weights)
        network_biases.append(biases)

    time_start = time.time()
    with torch.no_grad():
        network = NetworkModule(n_vars, network_weights, network_biases, network_size)
        res_under, res_over = network(inputs)

    for ru, ro in zip(res_under, res_over):
        ru = torch.reshape(ru, (ru.size(0) // network.n_vars, network.n_vars, ru.size(1)))
        ro = torch.reshape(ro, (ro.size(0) // network.n_vars, network.n_vars, ro.size(1)))
        l = ibf_minmax_cpp.ibf_minmax(ru)[0]
        u = ibf_minmax_cpp.ibf_minmax(ro)[1]

    print(f'lower: {l}, upper: {u}')
    time_end = time.time()
    print('time ', time_end - time_start)

if __name__ == '__main__':
    main()
