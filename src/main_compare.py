import torch
import time


### import BERN-IBF-NN files
from network_modules_list import *
from relu_coeffs import *
from poly_utils import *

### import BERN-NN-Valen files
import sys
sys.path.append('/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/BERN-NN-Valen')
# from torch_lists_modules import NetworkModule, generate_input, convert_to_dense

import sys
sys.path.append('/bern_cuda_ext')
import ibf_minmax_cpp

### import BERN-NN files
from torch_modules import NetworkModule as OldNetworkModule, bern_coeffs_inputs




# # Set the default tensor type to be 32-bit float on the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# # set a random seed
torch.manual_seed(0)

### 1. Define the network parameters
n_vars = 5
intervals = [[-1, 1] for i in range(n_vars)]
intervals = torch.tensor(intervals, dtype = torch.float32)
inputs = generate_inputs(n_vars, intervals, device = 'cuda')
network_size = torch.tensor([n_vars, 5, 1])
network_weights = []
network_biases = []

for i in range(len(network_size) - 1):
    weights = torch.randn(network_size[i  + 1], network_size[i])
    biases = torch.randn(network_size[i  + 1], 1)
    network_weights.append(weights)
    network_biases.append(biases)

### 2. apply BERN-IBF-NN and compute the results and time
time_start = time.time()
with torch.no_grad():
    with torch.cuda.device(0):
        network = NetworkModule(n_vars, network_weights, network_biases, network_size)
        res_under, res_over = network(inputs)

res_under = torch.reshape(res_under[0], (res_under[0].size(0) // n_vars, n_vars, res_under[0].size(1)))
l = ibf_minmax_cpp.ibf_minmax(res_under)[0]
time_end = time.time()
print('time BERN-IBF-NN ', time_end - time_start)
print(l)


# ### 3. apply BERN-NN-Valen and compute the results and time
order = 2
linear_iter_numb = 0
gpus = 1

### transpose the network weights 
network_weights = [w.t() for w in network_weights]
print(network_size)
network_module = OldNetworkModule(n_vars, intervals, network_weights, network_biases, network_size, order, linear_iter_numb, gpus)

print("OLD FORMAT")
inputs = bern_coeffs_inputs(n_vars, intervals)
time_start = time.time()
with torch.no_grad():
    with torch.cuda.device(0):
        result_under_orig, result_over_orig, network_nodes_des = network_module(inputs)

time_end = time.time()
print('time BERN-NN', time_end - time_start )
print(torch.min(result_over_orig))


