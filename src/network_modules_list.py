import torch
from poly_utils import *
from relu_coeffs import *
import time

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
def generate_inputs(n_vars, intervals, device = 'cuda'):
    inputs = []
    for i in range(n_vars):
        ith_input = torch.ones((n_vars, 2), dtype = torch.float32, device = device)
        ith_input[i, :] = intervals[i]
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
def ibf_tensor_prod_input_weights(n_vars, inputs_degrees, inputs, weights):
    
    ### if all the weights are zeros then node_input = torch.zeros(n_vars, 1) and degree_node_input = torch.zeros(n_vars)
    if torch.all(weights == 0):
        return torch.zeros(n_vars, 1), torch.zeros(n_vars)

 
    ### multiply the first input with the first weight if both are not zero
    if torch.all(inputs[0] == 0) or torch.all(weights[0] == 0):
        node_input = torch.zeros(n_vars, 1)
        degree_node_input = torch.zeros(n_vars)
    else:
        ### get the degree of the node_input
        degree_node_input = inputs_degrees[0, :]
        node_input = mult_with_constant(n_vars, inputs[0], inputs[0].size(0) // n_vars, degree_node_input, weights[0])
    for i in range(1, len(inputs)):
        ### multiply each ith input with the ith weight if both are not zero
        if torch.all(inputs[i] == 0) or torch.all(weights[i] == 0):
            continue
        ### multiply the ith input with the ith weight
        else:
            ith_input_weight = mult_with_constant(n_vars, inputs[i], inputs[i].size(0) // n_vars, inputs_degrees[i, :], weights[i])
            ### add the ith input to the node_input
            # print('degree_node_input ', degree_node_input)
            # print('inputs_degrees[i, :] ', inputs_degrees[i, :])
            # print('node_input ', node_input)
            # print('ith_input_weight ', ith_input_weight)
            node_input = sum_2_polys(n_vars, node_input, node_input.size(0) // n_vars, degree_node_input, ith_input_weight, ith_input_weight.size(0) // n_vars, inputs_degrees[i, :])
            ### update the degree of the node_input by taking the maximum wise between the degree of the node_input and the degree of the ith_input_weight
            degree_node_input = torch.max(degree_node_input, inputs_degrees[i, :])
    
    return node_input, degree_node_input






###################################################
### class NodeModule() takes as inputs:
### input_under and input_over and their degree
### and pass them through he node quadratic bounds
### q_under(input_under); q_over(input_over)
###################################################
class NodeModule(torch.nn.Module):

    def __init__(self, n_vars):
        super().__init__()
        self.n_vars = n_vars


    def forward(self, input_under, input_over, input_under_degree, input_over_degree):
        # ### clone input_under and input_over
        # input_under_clone = input_under.clone()
        # input_over_clone = input_over.clone()
        # print(input_under_clone)
        input_under = torch.reshape(input_under, (input_under.size(0) // self.n_vars, self.n_vars, input_under.size(1)))
        input_over = torch.reshape(input_over, (input_over.size(0) // self.n_vars, self.n_vars, input_over.size(1)))
        ### get the bounds for the node's relu:  l = min(input_under) and u = max(input_over)
        # print(input_under_clone)
        l = ibf_minmax_cpp.ibf_minmax(input_under)[0]
        u = ibf_minmax_cpp.ibf_minmax(input_over)[1]


        input_under = torch.reshape(input_under, (input_under.size(0) * self.n_vars, input_under.size(2)))
        input_over = torch.reshape(input_over, (input_over.size(0) * self.n_vars, input_over.size(2)))
        

        # print('l, u ', l, u)

        
        
        ### if u <= 0: return relu_node_under = relu_node_over = torch.tensor([0.]), and their degrees = torch.zeros(self.n_vars)
        if u <= 0.0:
            return torch.tensor([0.]), torch.tensor([0.]), torch.zeros(self.n_vars), torch.zeros(self.n_vars)
        ### if l >= 0: return relu_node_under = input_under and relu_node_over = input_over and their degrees = input_under_degree and input_over_degree
        elif l >= 0.0:
            return input_under, input_over, input_under_degree, input_over_degree
        
        ### if l < 0 and u > 0: compute the relu_node_under and relu_node_over
        else:
            bounds = torch.tensor([l, u])

            ### compute quadratic upper and lower polynomials coefficients from bounds = [l, u]
            coeffs_obj = relu_monom_coeffs(bounds)
            relu_coeffs_under = coeffs_obj.get_monom_coeffs_under(bounds)
            relu_coeffs_over = coeffs_obj.get_monom_coeffs_over(bounds)
            # print('relu_coeffs_under ', relu_coeffs_under)
            # print('relu_coeffs_over ', relu_coeffs_over)
            ### if relu_coeffs_under is all zeros then make relu_node_under = torch.empty(0) and pass the node's inputs input_over through the node's quadratic polynomials coefficents  relu_coeffs_over
            if torch.all(relu_coeffs_under == 0):
                # print('all relu_coeffs_under are zeros')
                relu_node_under = torch.tensor([0.])
                ### now keep the same next steps for relu_node_over
                ### pass the node's inputs input_over through the node's quadratic polynomials coefficents  relu_coeffs_over 
                # print(input_under)
                # print(input_under.size(0))
                num_terms_input_over = input_over.size(0) // self.n_vars
                relu_node_over = quad_of_poly(self.n_vars, input_over, num_terms_input_over, input_over_degree, relu_coeffs_over)
                ### compute the degree of relu_node_over = 2.0 * input_over_degree
                input_over_degree = 2.0 * input_over_degree
                return relu_node_under, relu_node_over, torch.zeros(self.n_vars), input_over_degree
            
            ### if [a, b, c] == [0, 1, 0] then make relu_node_under = input_under and input_under_degree and pass the node's inputs input_over through the node's quadratic polynomials coefficents  relu_coeffs_over
            elif relu_coeffs_under[1] == 1.0:
                ### now keep the same next steps for relu_node_over
                ### pass the node's inputs input_over through the node's quadratic polynomials coefficents  relu_coeffs_over 
                # print(input_under)
                # print(input_under.size(0))
                num_terms_input_over = input_over.size(0) // self.n_vars
                relu_node_over = quad_of_poly(self.n_vars, input_over, num_terms_input_over, input_over_degree, relu_coeffs_over)
                ### compute the degree of relu_node_over = 2.0 * input_over_degree
                input_over_degree = 2.0 * input_over_degree
                return input_under, relu_node_over, input_under_degree, input_over_degree
                
            else:
                ### pass the node's inputs input_under and input_over through the node's quadratic polynomials coefficents  relu_coeffs_over and relu_coeffs_under
                num_terms_input_under = input_under.size(0) // self.n_vars
                num_terms_input_over = input_over.size(0) // self.n_vars
                # print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
                # print('num_terms_input_under ', num_terms_input_under)
                relu_node_under = quad_of_poly(self.n_vars, input_under, num_terms_input_under, input_under_degree, relu_coeffs_under)
                # print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
                relu_node_over = quad_of_poly(self.n_vars, input_over, num_terms_input_over, input_over_degree, relu_coeffs_over)
                # print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
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

    def __init__(self, n_vars, layer_weights, layer_biases, layer_size):
        super().__init__()
        self.n_vars = n_vars
        self.layer_size = layer_size
        ### get the psoitive and negative weights from the layer_weights
        self._layer_weights_pos = torch.nn.functional.relu(layer_weights)
        self._layer_weights_neg = (-1) * torch.nn.functional.relu((-1) * layer_weights)
        self.layer_biases = layer_biases
        ### call self.node object
        self.node = NodeModule(self.n_vars)

    def forward(self, layer_inputs_under, layer_inputs_over, layer_inputs_under_degrees, layer_inputs_over_degrees, activation):

        ### for each node in the layer: propagate layer_inputs_under and layer_inputs_over through the node's weights and pass them through the node 
        ### TO DO: parallize this operation by batches of nodes
        results_under = []
        results_over = []
        results_under_degrees = []
        results_over_degrees = []
        # print('layer_size ', self.layer_size)
        for i in range(self.layer_size):
            print('########################## node number ', i, ' ##########################')
            ### get node 's lower input node_under 
            # print('layer_inputs_over_degrees ', layer_inputs_over_degrees)
            # print('layer_inputs_over ', layer_inputs_over)
            # print('self._layer_weights_neg[i] ', self._layer_weights_neg[i])
            combined_node_under_1, combined_node_under_1_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_over_degrees, layer_inputs_over, self._layer_weights_neg[i, :])
            # print('layer_inputs_under ', layer_inputs_under)
            # print('self._layer_weights_pos[i] ', self._layer_weights_pos[i, :])
            # print('layer_inputs_under_degrees ', layer_inputs_under_degrees)
            combined_node_under_2, combined_node_under_2_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_under_degrees, layer_inputs_under, self._layer_weights_pos[i, :])
                # print('size of combined_node_under_1 ', combined_node_under_1.size())
                # print('size of combined_node_under_2 ', combined_node_under_2.size())
   
            ### sum the two tensors combined_node_under_1 and combined_node_under_2
            # print('combined_node_under_1 ', combined_node_under_1)
            # print('combined_node_under_2 ', combined_node_under_2)
            node_under = sum_2_polys(self.n_vars, combined_node_under_1, combined_node_under_1.size(0) // self.n_vars, combined_node_under_1_degree, combined_node_under_2, combined_node_under_2.size(0) // self.n_vars, combined_node_under_2_degree)
            ### get node_under's degree
            node_under_degree = torch.max(combined_node_under_1_degree, combined_node_under_2_degree)
            ### add the bias to node_under
            node_under = add_with_constant(self.n_vars, node_under, self.layer_biases[i])

            ### get node 's upper input node_over
            combined_node_over_1, combined_node_over_1_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_over_degrees, layer_inputs_over, self._layer_weights_pos[i])
            combined_node_over_2, combined_node_over_2_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_under_degrees, layer_inputs_under, self._layer_weights_neg[i])
            ### sum the two tensors combined_node_over_1 and combined_node_over_2
            node_over = sum_2_polys(self.n_vars, combined_node_over_1, combined_node_over_1.size(0) // self.n_vars, combined_node_over_1_degree, combined_node_over_2, combined_node_over_2.size(0) // self.n_vars, combined_node_over_2_degree)
            ### get node_over's degree
            node_over_degree = torch.max(combined_node_over_1_degree, combined_node_over_2_degree)
            ### add the bias to node_over
            node_over = add_with_constant(self.n_vars, node_over, self.layer_biases[i])

            ### get node 's lower and upper outputs node_output_under node_output_over
            # print(node_under)
            # print(node_over)
            print('nodebeforenodebeforenodebeforenodebeforenodebeforenodebeforenodebeforenodebeforenodebeforenodebeforenodebeforenodebeforenodebeforenodebeforenodebefore')
            print('node_under_size ', node_under.size())
            print('node_over_size ', node_over.size())
            if activation == 'relu':
                node_output_under, node_output_over, node_output_under_degree, node_output_over_degree = self.node(node_under, node_over, node_under_degree, node_over_degree)
            else:
                node_output_under = node_under
                node_output_over = node_over
                node_output_under_degree = node_under_degree
                node_output_over_degree = node_over_degree
            # print('nodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafternodeafter')
            # print('node_output_under_size ', node_output_under.size())
            # print('node_output_over_size ', node_output_over.size())
            ### append the results in results_under and results_over
            results_under.append(node_output_under)
            results_over.append(node_output_over)
            results_under_degrees.append(node_output_under_degree)
            results_over_degrees.append(node_output_over_degree)

            ### return the results in results_under and results_over and their degrees
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

    def __init__(self, n_vars, network_weights, network_biases, network_size):
        super().__init__()
        self.layers = [LayerModule(n_vars, network_weights[i], network_biases[i], network_size[i + 1]) for i in range(len(network_size) - 1) ]


    ### propagate the inputs_over and inputs_under through each layer in the network
    def forward(self, inputs):
        inputs_under = inputs
        inputs_over = inputs
        ### inputs_under_degrees = inputs_over_degrees = torch.diag(n_vars)
        inputs_under_degrees = torch.ones(n_vars, n_vars)
        inputs_over_degrees = torch.ones(n_vars, n_vars)
        i = 0
        for layer in self.layers:
            print('################################################# layer number ', i, ' #################################################')
            # print('inputs_under ', inputs_under)
            # print('inputs_over ', inputs_over)
            # print('inputs_under_degrees ', inputs_under_degrees)
            # print('inputs_over_degrees ', inputs_over_degrees)
            if i == len(self.layers) - 1:
                inputs_under, inputs_over, inputs_under_degrees, inputs_over_degrees = layer(inputs_under, inputs_over, inputs_under_degrees, inputs_over_degrees, 'linear')
            else:
                inputs_under, inputs_over, inputs_under_degrees, inputs_over_degrees  = layer(inputs_under, inputs_over, inputs_under_degrees, inputs_over_degrees, 'relu')
            i += 1
        return inputs_under, inputs_over




### testing the NodeModule, LayerModule, and NetworkModule classes

if __name__ == "__main__":
    # # Set the default tensor type to be 32-bit float on the GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # # set a random seed
    torch.manual_seed(0)


    # ### 1) test generate_inputs function 
    # n_vars = 4
    # intervals = torch.tensor([[1., 2.], [3., 4.], [7., 9.], [5., 6.]])
    # res = generate_inputs(n_vars, intervals, device = 'cuda')
    # print(res)

    ### 2) test ibf_tensor_prod_input_weights function
    # n_vars = 4
    # intervals = torch.tensor([[1., 2.], [3., 4.], [7., 9.], [5., 6.]])
    # inputs = generate_inputs(n_vars, intervals, device = 'cuda')
    # print(inputs)
    # weights = torch.tensor([[1.], [2.], [3.], [5.]])
    # print(weights)
    # res = ibf_tensor_prod_input_weights(n_vars, inputs, weights)
    # print(res)


    # ### 3) test the NodeModule() class
    # n_vars = 4
    # degree = 1
    # intervals = torch.tensor([[1., 2.], [3., 4.], [7., 9.], [5., 6.]])
    # inputs = generate_inputs(n_vars, intervals, device = 'cuda')
    # print(inputs)
    # input_under = input_over = inputs
    # print(input_under.size(0) // n_vars)
    # node = NodeModule(n_vars, degree)
    # res_under, res_over = node(input_under, input_over)
    # print(res_under.size(0) // n_vars)

    # ### 4) tes LayerModule() class

    # n_vars = 4
    # degree = 1
    # intervals = torch.tensor([[1., 2.], [3., 4.], [7., 9.], [5., 6.]])
    # inputs = generate_inputs(n_vars, intervals, device = 'cuda')
    # layer_size = 5
    # layer_weights = torch.randn(layer_size, n_vars)
    # layer_biases = torch.randn(layer_size, 1)
    # input_under = input_over = inputs
    # layer = LayerModule(n_vars, degree, layer_weights, layer_biases, layer_size)
    # res_under, res_over = layer(input_under, input_over)
    # print(res_under)
    

    ### 5) tes NetworkModule() class

    n_vars = 3
    intervals = [[-1, 1] for i in range(n_vars)]
    intervals = torch.tensor(intervals, dtype = torch.float32)
    inputs = generate_inputs(n_vars, intervals, device = 'cuda')
    # print(inputs)
    ### convert inputs to a sparse representation
    # inputs = inputs.to_sparse()
    network_size = [n_vars, 4, 4, 4,  1]
    network_weights = []
    network_biases = []

    for i in range(len(network_size) - 1):
        weights = torch.randn(network_size[i  + 1], network_size[i])
        # print('weights ', weights)
        biases = torch.randn(network_size[i  + 1], 1)
        network_weights.append(weights)
        network_biases.append(biases)

    # network_weights = torch.tensor(network_weights)
    # network_biases =  torch.tensor(network_biases)   
    time_start = time.time()
    with torch.no_grad():
        network = NetworkModule(n_vars, network_weights, network_biases, network_size)
        res_under, res_over = network(inputs)
    time_end = time.time()
    print('time ', time_end - time_start)
    # print('res_under', res_under)









