import time
import torch
import sys
sys.path.append('./rep/')

from poly_utils_cuda import *
from relu_coeffs import *
from lower_upper_linear_ibf import *
from rep.Bern_NN_IBF import NetworkModule 

def generate_inputs(n_vars, intervals, device = 'cuda'):
    inputs = []
    for i in range(n_vars):
        ith_input = torch.ones((n_vars, 2), dtype = torch.float32, device = device)
        ith_input[i, :] = intervals[i]
        inputs.append(ith_input)
    return inputs

def create_network(network_size):
    network_weights = []
    network_biases = []

    for i in range(len(network_size) - 1):
        weights = torch.ones(network_size[i  + 1], network_size[i])
        biases = torch.zeros(network_size[i  + 1], 1)
        network_weights.append(weights)
        network_biases.append(biases)
    return network_weights, network_biases

##################################################
# Build a simple MLP. We assume that each linear
# layer is followed by a ReLU activation function.
##################################################

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.manual_seed(0)

# The number of input variables essentially defines the
# size of the dimension of the input space.
n_vars = 5

# The intervals sets the extent of each dimension.
# (In this case, the input is the five dimensional hypercube [-1, 1]^5)
intervals = [[-1, 1] for i in range(n_vars)]
intervals = torch.tensor(intervals, dtype = torch.float32)

# Create a small four layer network, where each layer
# has five neurons. We currently set all of the weights
# to 1 (rather than being randomized), so it is easier 
# to see how the bounds change with different settings. 
network_size = [n_vars, 5, 5, 5, 1]
network_weights, network_biases = create_network(network_size)

# We can optionally linearize after every N-th layer. There is
# a trade-off between performance and the quality of bound.
# This setting will linearize the bound after every layer, 
# resuling in looser bounds compared to not applying linearization.
lin_itr_numb = 1

# The inputs to the network module are based on the intervals.
# These are used to propagate bounds in each layer.
inputs = generate_inputs(n_vars, intervals, device = 'cuda')

time_start = time.time()
with torch.no_grad():
    # Calling the forward() function on the NetworkModule computes the upper and lower bounds.
    network = NetworkModule(n_vars, intervals, network_weights, network_biases, network_size, lin_itr_numb)
    res_under, res_over = network(inputs)
time_end = time.time()
print('time ', time_end - time_start)
