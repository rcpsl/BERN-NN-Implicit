import torch
import torch.nn as nn
#from poly_utils_parallel import *
from poly_utils_cuda import *
from relu_coeffs import *
import time
import torch.multiprocessing as mp
import torch.distributed as dist

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:516"

import sys
sys.path.append('/bern_cuda_ext')
import ibf_minmax_cpp



###################################################
### create a 2D inputs of the input layer for the network
### of size(n * n, 2), where the i^th sub-tensor in inputs
### represent the input for the i^th node, and is written as follows:
### inputs[i + n, :] = tensor([1, 0],
###                             .
###                       [dmin_i, dmax_i],
###                             .
###                           [1, 0]) 
### n_vars: number of variables
### intervals: = tensor([[dmin_0, dmax_0], ..., [dmin_n, dmax_n]]):
### domain for the NN's input
### move the inputs to the GPU cuda
###################################################
def generate_inputs(n_vars, intervals, device):
    inputs = []
    for i in range(n_vars):
        ith_input = torch.ones((n_vars, 2), dtype = torch.float32, device = device)
        ith_input[i, :] = intervals[i].to(device)
        inputs.append(ith_input)
    return inputs


###################################################
### multiply the the tensor of inputs with
### the 1D weights by multiplying each component 
### of inputs with each component of weights
### and perform degree elevation when summing the terms
### n_vars: number of variables
### inputs_degrees: the degrees of the inputs: 2D tensor of size (num_nodes, n_vars)
### inputs
###################################################
def ibf_tensor_prod_input_weights(n_vars, inputs_degrees, inputs, weights, device):
    # Ensure inputs are on the correct device
    inputs = [input_.to(device) for input_ in inputs]
    weights = weights.to(device)
    inputs_degrees = inputs_degrees.to(device)
    
    # If all the weights are zeros then return tensors of zeros on the correct device
    if torch.all(torch.abs(weights) == 0):
        degree_node_input = torch.zeros(n_vars, device=device)
        return torch.zeros((n_vars, 1), device=device), degree_node_input

    # Multiply the first input with the first weight if both are not zero
    if torch.all(torch.abs(inputs[0]) == 0) or torch.all(torch.abs(weights[0]) == 0):
        degree_node_input = torch.zeros(n_vars, device=device)
        node_input = torch.zeros((n_vars, 1), device=device)
    else:
        # Get the degree of the node_input
        degree_node_input = inputs_degrees[0, :]   
        node_input = mult_with_constant(n_vars, inputs[0], weights[0])
    
    for i in range(1, len(inputs)):
        if torch.all(torch.abs(weights[i]) != 0):
            ith_input_weight = mult_with_constant(n_vars, inputs[i], weights[i])
            node_input = sum_2_polys(n_vars, node_input, node_input.size(0) // n_vars, degree_node_input, ith_input_weight, ith_input_weight.size(0) // n_vars, inputs_degrees[i, :])
            # Update the degree of the node_input by taking the max wise between the degree of the node_input and the degree of the ith_input_weight
            degree_node_input = torch.max(degree_node_input, inputs_degrees[i, :])

    return node_input, degree_node_input







###################################################
### class NodeModule() takes as inputs:
### input_under and input_over and their degree
### and pass them through he node quadratic bounds
### q_under(input_under); q_over(input_over)
###################################################
class NodeModule(torch.nn.Module):

    def __init__(self, n_vars, node_pos_weights, node_neg_weights, node_bias, activation, device):
        super(NodeModule, self).__init__()
        self.n_vars = n_vars
        self.device = device
        self.node_pos_weights = node_pos_weights.to(device)
        self.node_neg_weights = node_neg_weights.to(device)
        self.node_bias = node_bias.to(device)
        self.activation = activation


    def forward(self, layer_inputs_under, layer_inputs_over, layer_inputs_under_degrees, layer_inputs_over_degrees):
        # Convert inputs and degrees to the module's device
        layer_inputs_under = [input_.to(self.device) for input_ in layer_inputs_under]
        layer_inputs_over = [input_.to(self.device) for input_ in layer_inputs_over]
        layer_inputs_under_degrees = layer_inputs_under_degrees.to(self.device)
        layer_inputs_over_degrees = layer_inputs_over_degrees.to(self.device)

            
        ### for each node in the layer: propagate layer_inputs_under and layer_inputs_over through the node's weights and pass them through the node
        ### get node 's lower input node_under 
        combined_node_under_1, combined_node_under_1_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_over_degrees, layer_inputs_over, self.node_neg_weights, self.device)
        combined_node_under_2, combined_node_under_2_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_under_degrees, layer_inputs_under, self.node_pos_weights, self.device)

        ### sum the two tensors combined_node_under_1 and combined_node_under_2
        input_under = sum_2_polys(self.n_vars, combined_node_under_1, combined_node_under_1.size(0) // self.n_vars, combined_node_under_1_degree, combined_node_under_2, combined_node_under_2.size(0) // self.n_vars, combined_node_under_2_degree)
        ### get input_under's degree
        input_under_degree = torch.max(combined_node_under_1_degree, combined_node_under_2_degree).to(self.device)
        ## delete combined_node_under_1 and combined_node_under_1_degree
        del combined_node_under_1
        del combined_node_under_1_degree
        ## delete combined_node_under_2 and combined_node_under_2_degree
        del combined_node_under_2
        del combined_node_under_2_degree
        ### add the bias to input_under
        input_under = add_with_constant(self.n_vars, input_under, self.node_bias)

        ### get node 's upper input input_over
        combined_node_over_1, combined_node_over_1_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_over_degrees, layer_inputs_over, self.node_pos_weights, self.device)
        combined_node_over_2, combined_node_over_2_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_under_degrees, layer_inputs_under, self.node_neg_weights, self.device)
        ### sum the two tensors combined_node_over_1 and combined_node_over_2
        input_over = sum_2_polys(self.n_vars, combined_node_over_1, combined_node_over_1.size(0) // self.n_vars, combined_node_over_1_degree, combined_node_over_2, combined_node_over_2.size(0) // self.n_vars, combined_node_over_2_degree)
        ### get input_over's degree
        input_over_degree = torch.max(combined_node_over_1_degree, combined_node_over_2_degree).to(self.device)
        ## delete combined_node_over_1 and combined_node_over_1_degree
        del combined_node_over_1
        del combined_node_over_1_degree
        ## delete combined_node_over_2 and combined_node_over_2_degree
        del combined_node_over_2
        del combined_node_over_2_degree
        ### add the bias to node_over
        input_over = add_with_constant(self.n_vars, input_over, self.node_bias)


        if self.activation == 'linear':
            return input_under, input_over, input_under_degree, input_over_degree

        if self.activation == 'relu':
            
            input_under = torch.reshape(input_under, (input_under.size(0) // self.n_vars, self.n_vars, input_under.size(1)))
            input_over = torch.reshape(input_over, (input_over.size(0) // self.n_vars, self.n_vars, input_over.size(1)))
            ### get the bounds for the node's relu:  l = min(input_under) and u = max(input_over)
            l = ibf_minmax_cpp.ibf_minmax(input_under)[0]
            u = ibf_minmax_cpp.ibf_minmax(input_over)[1]
            input_under = torch.reshape(input_under, (input_under.size(0) * self.n_vars, input_under.size(2)))
            input_over = torch.reshape(input_over, (input_over.size(0) * self.n_vars, input_over.size(2)))
                        
            ### if u <= 0: return relu_node_under = relu_node_over = torch.tensor([0.]), and their degrees = torch.zeros(self.n_vars)
            if u <= 0.0:
                return torch.tensor([0.]).to(self.device), torch.tensor([0.]).to(self.device), torch.zeros(self.n_vars).to(self.device), torch.zeros(self.n_vars).to(self.device)
            ### if l >= 0: return relu_node_under = input_under and relu_node_over = input_over and their degrees = input_under_degree and input_over_degree
            elif l >= 0.0:
                return input_under, input_over, input_under_degree, input_over_degree
            
            ### if l < 0 and u > 0: compute the relu_node_under and relu_node_over
            else:
                bounds = torch.tensor([l, u]).to(self.device)

                ### compute quadratic upper and lower polynomials coefficients from bounds = [l, u]
                coeffs_obj = relu_monom_coeffs(bounds)
                relu_coeffs_under = coeffs_obj.get_monom_coeffs_under(bounds)
                relu_coeffs_over = coeffs_obj.get_monom_coeffs_over(bounds)
                ### if relu_coeffs_under is all zeros then make relu_node_under = torch.empty(0) and pass the node's inputs input_over through the node's quadratic polynomials coefficents  relu_coeffs_over
                if torch.all(relu_coeffs_under == 0):
                    relu_node_under = torch.zeros(self.n_vars, 1, self.device)
                    ### now keep the same next steps for relu_node_over
                    ### pass the node's inputs input_over through the node's quadratic polynomials coefficents  relu_coeffs_over 
                    num_terms_input_over = input_over.size(0) // self.n_vars
                    relu_node_over = quad_of_poly(self.n_vars, input_over, num_terms_input_over, input_over_degree, relu_coeffs_over)
                    ### compute the degree of relu_node_over = 2.0 * input_over_degree
                    input_over_degree = 2.0 * input_over_degree
                    return relu_node_under, relu_node_over, torch.zeros(self.n_vars, self.device), input_over_degree
                
                ### if [a, b, c] == [0, 1, 0] then make relu_node_under = input_under and input_under_degree and pass the node's inputs input_over through the node's quadratic polynomials coefficents  relu_coeffs_over
                elif relu_coeffs_under[1] == 1.0:
                    ### now keep the same next steps for relu_node_over
                    ### pass the node's inputs input_over through the node's quadratic polynomials coefficents  relu_coeffs_over 
                    num_terms_input_over = input_over.size(0) // self.n_vars
                    relu_node_over = quad_of_poly(self.n_vars, input_over, num_terms_input_over, input_over_degree, relu_coeffs_over)
                    ### compute the degree of relu_node_over = 2.0 * input_over_degree
                    input_over_degree = 2.0 * input_over_degree
                    return input_under, relu_node_over, input_under_degree, input_over_degree
                    
                else:
                    ### pass the node's inputs input_under and input_over through the node's quadratic polynomials coefficents  relu_coeffs_over and relu_coeffs_under
                    num_terms_input_under = input_under.size(0) // self.n_vars
                    num_terms_input_over = input_over.size(0) // self.n_vars
                    relu_node_under = quad_of_poly(self.n_vars, input_under, num_terms_input_under, input_under_degree, relu_coeffs_under)
                    relu_node_over = quad_of_poly(self.n_vars, input_over, num_terms_input_over, input_over_degree, relu_coeffs_over)
                    return relu_node_under, relu_node_over, 2 * input_under_degree , 2 * input_over_degree


###################################################
### class LayerModule() takes as inputs:
### layer_inputs_under and layer_inputs_over and 
### and propagate them through the layer. 
### n_vars: number of variables
### degree: degree for layer_inputs_under and layer_inputs_over
### layer_weights: layer's weights where the ith row
### for the layer_weights corresponds to the weights 
### for the ith node for that layer.
### layer_biases: the biases for the layer
### layer_size: the number of nodes for the layer
###################################################
class LayerModule(torch.nn.Module):

    def __init__(self, n_vars, layer_weights, layer_biases, layer_size, activation, device):
        super().__init__()
        self.n_vars = n_vars
        self.layer_size = layer_size
        self._layer_weights_pos = torch.nn.functional.relu(layer_weights).to(device)
        self._layer_weights_neg = (-1) * torch.nn.functional.relu((-1) * layer_weights).to(device)
        self.layer_biases = layer_biases.to(device)
        self.activation = activation
        self.device = device  # Added device attribute
        # Nodes will be initialized in the forward pass to ensure correct device allocation
        self.nodes = [NodeModule(self.n_vars, self._layer_weights_pos[i, :], self._layer_weights_neg[i, :], self.layer_biases[i], self.activation, device) for i in range(self.layer_size)]

        # Set mp start method to spawn to avoid issues on CUDA
        #mp.set_start_method('spawn', force=True)

    def forward(self,
                layer_inputs_under,
                layer_inputs_over,
                layer_inputs_under_degrees,
                layer_inputs_over_degrees,
                rank,
                size):
        num_gpus = size

        # Determine which nodes the current rank will process 
        chunk_size = self.layer_size // num_gpus
        chunk_start = rank * chunk_size
        chunk_end = chunk_start + chunk_size
        if rank == size - 1:
            chunk_end = self.layer_size
        
        gpu_batch_indices = list(range(chunk_start, chunk_end))
        print(f'{rank}: {gpu_batch_indices}')

        # these results are local to the current device
        local_results_under = []
        local_results_over = []
        local_results_under_degrees = []
        local_results_over_degrees = []

        for node_idx in gpu_batch_indices:
            node = self.nodes[node_idx]
            (node_output_under, node_output_over, node_output_under_degree, node_output_over_degree) = node(
                layer_inputs_under, layer_inputs_over, layer_inputs_under_degrees, layer_inputs_over_degrees)

            local_results_under.append(node_output_under)
            local_results_over.append(node_output_over)
            local_results_under_degrees.append(node_output_under_degree)
            local_results_over_degrees.append(node_output_over_degree)

        # Gather all of the local results, so each GPU has the same copy of results_*
        # Note: all_gather_object results in GPU -> CPU transfer for pickling
        # TODO: it may makes sense to combine these into one all_gather_object call.
        results_under = [None for _ in range(num_gpus)]
        dist.all_gather_object(results_under, local_results_under)

        results_over = [None for _ in range(num_gpus)]
        dist.all_gather_object(results_over, local_results_over)

        results_under_degrees = [None for _ in range(num_gpus)]
        dist.all_gather_object(results_under_degrees, local_results_under_degrees)

        results_over_degrees = [None for _ in range(num_gpus)]
        dist.all_gather_object(results_over_degrees, local_results_over_degrees)

        # flatten the List[List[Tensor]] to List[Tensor]
        # for some reason, the Tensors may still be on the original device...
        # so explicitly moving them to the current device seems necessary
        results_under = [r.to(self.device) for rs in results_under for r in rs]
        results_over = [r.to(self.device) for rs in results_over for r in rs]
        results_under_degrees = [r.to(self.device) for rs in results_under_degrees for r in rs]
        results_over_degrees = [r.to(self.device) for rs in results_over_degrees for r in rs]

        return results_under, results_over, torch.stack(results_under_degrees), torch.stack(results_over_degrees)


###################################################
### class NetworkModule() takes as inputs:
### layer_inputs_under and layer_inputs_over and 
### and propagate them through the layer. 
### n_vars: number of variables
### degree: degree for layer_inputs_under and layer_inputs_over
### layer_weights: layer's weights where the ith row
### for the layer_weights corresponds to the weights 
### for the ith node for that layer.
### layer_biases: the biases for the layer
### layer_size: the number of nodes for the layer
###################################################
class NetworkModule(torch.nn.Module):
    def __init__(self, n_vars, network_weights, network_biases, network_size, device):
        super().__init__()
        self.n_vars = n_vars
        self.device = device
        ### initialize the layers with the weights and biases and their sizes and the activation function
        self.layers = [
            LayerModule(n_vars, network_weights[i], network_biases[i], network_size[i + 1], 'relu', device)
            if i != len(network_size) - 2 else 
            LayerModule(n_vars, network_weights[i], network_biases[i], network_size[i + 1], 'linear', device)
            for i in range(len(network_size) - 1)
        ]

    ### propagate the inputs_over and inputs_under through each layer in the network
    def forward(self, inputs, rank, size):
        inputs_under = inputs
        inputs_over = inputs
        ### Initialize degrees to identity since we're working with linear functions initially
        inputs_under_degrees = torch.ones(self.n_vars, self.n_vars, device=self.device)
        inputs_over_degrees = torch.ones(self.n_vars, self.n_vars, device=self.device)
        
        for i, layer in enumerate(self.layers):
            print(f'################################################# Layer number {i} #################################################')
            inputs_under, inputs_over, inputs_under_degrees, inputs_over_degrees = layer(
                inputs_under, inputs_over, inputs_under_degrees, inputs_over_degrees, rank, size
            )
        
        return inputs_under, inputs_over

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def run(rank, size):
    assert rank < size

    # Setting process device is necessary for dist.all_gather_object
    torch.cuda.set_device(f'cuda:{rank}')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # each process uses the same starting seed, so they generate
    # the same model
    torch.manual_seed(0)
    
    device = f'cuda:{rank}'

    n_vars = 24
    intervals = [[-1, 1] for i in range(n_vars)]
    intervals = torch.tensor(intervals, dtype=torch.float32).to(device)
    inputs = generate_inputs(n_vars, intervals, device=device)
    network_size = [n_vars, 8, 8]
    network_size = torch.tensor(network_size).to(device)
    network_weights = []
    network_biases = []

    for i in range(len(network_size) - 1):
        weights = torch.randn(network_size[i + 1], network_size[i], device=device)
        biases = torch.randn(network_size[i + 1], 1, device=device)
        network_weights.append(weights)
        network_biases.append(biases)
  
    time_start = time.time()
    with torch.no_grad():
        network = NetworkModule(n_vars, network_weights, network_biases, network_size, device=device)
        res_under, res_over = network(inputs, rank, size)
    time_end = time.time()
    print('Time elapsed:', time_end - time_start)

    if rank == 0:
        print(res_under)
        print(res_over)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
