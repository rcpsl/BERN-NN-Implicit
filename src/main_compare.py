import torch
import time


### import BERN-IBF-NN files
from network_modules_over_GPUs import *
# from network_modules_batches import *
from relu_coeffs import *
# from poly_utils_old import *
from poly_utils_cuda import *
from test_poly_operations import *

### import BERN-NN-Valen files
import sys
sys.path.append('/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/BERN-NN-Valen')
# from torch_lists_modules import NetworkModule, generate_input, convert_to_dense

import sys
sys.path.append('/bern_cuda_ext')
import ibf_minmax_cpp

### import BERN-NN files
from torch_modules import NetworkModule as OldNetworkModule, bern_coeffs_inputs
from relu_bern_coeffs import Network




# # Set the default tensor type to be 32-bit float on the GPU
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# # set a random seed
# torch.manual_seed(0)
np.random.seed(0)

### 1. Define the network parameters
n_vars = 10
intervals = [[-1, 1] for i in range(n_vars)]
network_size = [n_vars, 50, 50,  1]

### create the network weights and biases  
network_weights = []
network_biases = []

for i in range(len(network_size) - 1):
    weights = np.random.randn(network_size[i  + 1], network_size[i])
    # weights = np.ones((network_size[i  + 1], network_size[i]))
    biases = np.zeros((network_size[i  + 1], 1))
    network_weights.append(weights)
    network_biases.append(biases)

### apply true network and compute the results and time
activation = 'relu'
rect = np.array(intervals)
net = Network(network_size, activation, 2, 'rand', rect)

net._W = network_weights
net._b = network_biases
    
inputs_samples = []
for i in range(n_vars):
    x = np.random.uniform(intervals[i][0], intervals[i][1], size = 100000)
    x = x.reshape(-1, 1)
    inputs_samples.append(x)

    
# print(inputs_samples)
X = np.concatenate(inputs_samples, axis = 1)
# print('X shape', X.shape)
Y = net.forward_prop(X.T)

true_l_u = [ [min(Y[i, :]), max(Y[i, :])] for i in range(Y.shape[0])]
# true_l_u = [[min(Y[0, :]), max(Y[0, :])]]


    
true_l_u = np.array(true_l_u) 
print('TRUE Bounds', true_l_u)

### 2. apply BERN-IBF-NN and compute the results and time
intervals_torch = torch.tensor(intervals, dtype = torch.float32)
inputs = generate_inputs(n_vars, intervals_torch, device = 'cuda')
network_size_torch = torch.tensor(network_size, dtype = torch.int32)
network_weights_torch = [torch.tensor(w, dtype = torch.float32) for w in network_weights]
network_biases_torch = [torch.tensor(b, dtype = torch.float32) for b in network_biases]


time_start = time.time()
with torch.no_grad():
    with torch.cuda.device(0):
        # network = NetworkModule(n_vars, network_weights, network_biases, network_size)
        network = NetworkModule(n_vars, network_weights_torch, network_biases_torch, network_size_torch)
        res_under, res_over = network(inputs)

time_end = time.time()
print('time BERN-NN-IBF ', time_end - time_start)

# ### convert res_under and res_over to dense tensors
# print(res_under)
# res_under_dense = implicit_to_dense(res_under[0], res_under[0].size(0) // n_vars)
# res_over_dense = implicit_to_dense(res_over[0], res_over[0].size(0) // n_vars)
# print(torch.min(res_under_dense), torch.max(res_over_dense))

# print(res_under[0])
res_under = torch.reshape(res_under[0], (res_under[0].size(0) // n_vars, n_vars, res_under[0].size(1)))
res_over = torch.reshape(res_over[0], (res_over[0].size(0) // n_vars, n_vars, res_over[0].size(1)))
l = ibf_minmax_cpp.ibf_minmax(res_under)[0]
u = ibf_minmax_cpp.ibf_minmax(res_over)[1]
### print the bounds
print('BERN-NN-IBF Bounds', l, u)



# ### 3. apply BERN-NN-Valen and compute the results and time
order = 2
linear_iter_numb = 0
gpus = 1

### transpose the torch network weights 
network_weights_torch = [w.t() for w in network_weights_torch]
# print(network_size)
network_module = OldNetworkModule(n_vars, intervals_torch, network_weights_torch, network_biases_torch, network_size_torch, order, linear_iter_numb, gpus)
# network_module = OldNetworkModule(n_vars, intervals, network_weights, network_biases, network_size, order, linear_iter_numb, gpus)

print("OLD TOOL")
inputs = bern_coeffs_inputs(n_vars, intervals_torch)
time_start = time.time()
with torch.no_grad():
    with torch.cuda.device(0):
        result_under_orig, result_over_orig, network_nodes_des = network_module(inputs)

time_end = time.time()
print('time BERN-NN', time_end - time_start )
# print(result_under_orig)
print('BERN-NN Bounds', torch.min(result_under_orig), torch.max(result_over_orig))



