import torch
import sys
sys.path.append('../src')
sys.path.append('../src/bern_cuda_ext')
from poly_utils import *
from relu_coeffs import *
from network_modules_batches import *
import ibf_minmax_cpp
import numpy as np
import time
import matplotlib.pyplot as plt

def main():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(0)

    n_hidden_layers_list = np.arange(1, 6)
    n_trials = 5
    runtimes = np.empty((n_trials, n_hidden_layers_list.shape[0]))

    for idx, n_hidden_layers in enumerate(n_hidden_layers_list):
        print(n_hidden_layers)
        for trial in range(3): 
            get_time(n_hidden_layers)
        for trial in range(n_trials): 
            dur = get_time(n_hidden_layers)
            runtimes[trial, idx] = dur
    print(runtimes)
    plt.boxplot(runtimes, labels=n_vars_list)
    plt.savefig('hidden_vs_time.pdf')

def get_time(n_vars):
    start = time.time()
    exp(n_vars)
    torch.cuda.synchronize()
    end = time.time()
    return end - start

def exp(hidden_layers):
    n_vars = 2
    intervals = [[-1, 1] for i in range(n_vars)]
    #intervals = torch.tensor(intervals, dtype = torch.float32)
    #inputs = generate_inputs(n_vars, intervals, device = 'cuda')
    neurons = 20
    hidden_layers = [neurons for _ in range(hidden_layers)]
    network_size =  [n_vars] + hidden_layers + [1]
    network_weights = []
    network_biases = []

    for i in range(len(network_size) - 1):
        weights = torch.randn(network_size[i + 1], network_size[i])
        biases = torch.randn(network_size[i + 1], 1)
        network_weights.append(weights)
        network_biases.append(biases)

    ### 2. apply BERN-IBF-NN and compute the results and time
    intervals_torch = torch.tensor(intervals, dtype = torch.float32)
    inputs = generate_inputs(n_vars, intervals_torch, device = 'cuda')
    network_size_torch = torch.tensor(network_size, dtype = torch.int32)
    network_weights_torch = network_weights#[torch.tensor(w, dtype = torch.float32) for w in network_weights]
    network_biases_torch = network_biases #[torch.tensor(b, dtype = torch.float32) for b in network_biases]


    time_start = time.time()
    with torch.no_grad():
        with torch.cuda.device(0):
            # network = NetworkModule(n_vars, network_weights, network_biases, network_size)
            network = NetworkModule(n_vars, network_weights_torch, network_biases_torch, network_size_torch)
            res_under, res_over = network(inputs)

    time_end = time.time()
    print('time BERN-NN-IBF ', time_end - time_start)

    res_under = torch.reshape(res_under[0], (res_under[0].size(0) // n_vars, n_vars, res_under[0].size(1)))
    res_over = torch.reshape(res_over[0], (res_over[0].size(0) // n_vars, n_vars, res_over[0].size(1)))
    l = ibf_minmax_cpp.ibf_minmax(res_under)[0]
    u = ibf_minmax_cpp.ibf_minmax(res_over)[1]
    ### print the bounds
    print('BERN-NN-IBF Bounds', l, u)


    """
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
    """
    
if __name__ == '__main__':
    main()
