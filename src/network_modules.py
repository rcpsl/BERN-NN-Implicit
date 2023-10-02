import torch
from poly_utils import *
from relu_coeffs import *


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
    inputs = torch.zeros((n_vars, n_vars, 2), dtype = torch.float32, device = device)
    inputs[:, :, 0] = 1
    indices = torch.arange(n_vars)
    inputs[indices, indices, :] = intervals
    inputs = torch.reshape(inputs, (-1, inputs.size(2)))
    return inputs

###################################################
### multiply the the tensor of inputs with
### the 1D weights by multiplying each component 
### of inputs with each component of weights
###################################################
def ibf_tensor_prod_input_weights(n_vars, inputs, weights):
    """
    inputs is a set of polynomials in IBF format
    weights is the parameters for one neuron? 
        TODO: change to weights for a batch of neurons
    """

    assert inputs.dim() == 2, 'expects four tensor dims [polynomials, terms * n_vars, max_degree]'
    # assert inputs.size(0) // n_vars == weights.size(0), 'expects one polynomial for each weight'

    # ### reshape inputs to a 2D tensor
    # inputs = inputs.reshape(-1, inputs.size(2))
    # num_terms_tensor = torch.tensor([inputs[i].size(0) // n_vars])
    n_terms = inputs.size(0) // len(weights) // n_vars 

    # Repeat the weights for each term in the corresponding polynomial.
    rep = torch.repeat_interleave(weights, repeats = n_terms * inputs.size(1)  , dim=0)
    term_scale = rep.reshape(-1, inputs.size(1))
    # Overwrite into a tensor of ones. 
    ones = torch.ones_like(inputs, device=inputs.device)
    # print(ones)
    # print(term_scale)
    ones[torch.arange(0, inputs.size(0), step = n_vars), :] = term_scale

    # Element-wise product scales one vector of each term by corresponding weight.
    scaled = inputs * ones

    # Reshaping essentially concatenates all terms into one large polynomial.
    # This is the sum of the scaled polynomials
    return scaled.reshape(-1, inputs.size(1))





###################################################
### class NodeModule() takes as inputs:
### input_under and input_over and their degree
### and pass them through he node quadratic bounds
### q_under(input_under); q_over(input_over)
###################################################
class NodeModule(torch.nn.Module):

    def __init__(self, n_vars, degree):
        super().__init__()
        self.n_vars = n_vars
        self.degree = degree


    def forward(self, input_under, input_over):
        ### clone input_under and input_over
        input_under_clone = input_under.clone()
        input_over_clone = input_over.clone()
        # print(input_under_clone)
        input_under_clone = torch.reshape(input_under_clone, (input_under_clone.size(0) // self.n_vars, self.n_vars, input_under_clone.size(1)))
        input_over_clone = torch.reshape(input_over_clone, (input_over_clone.size(0) // self.n_vars, self.n_vars, input_over_clone.size(1)))
        ### get the bounds for the node's relu:  l = min(input_under) and u = max(input_over)
        # print(input_under_clone)
        l = ibf_minmax_cpp.ibf_minmax(input_under_clone)[0]
        u = ibf_minmax_cpp.ibf_minmax(input_over_clone)[1]

        bounds = torch.tensor([l, u])

        ### compute quadratic upper and lower polynomials coefficients from bounds = [l, u]
        coeffs_obj = relu_monom_coeffs(bounds)
        relu_coeffs_under = coeffs_obj.get_monom_coeffs_under(bounds)
        relu_coeffs_over = coeffs_obj.get_monom_coeffs_over(bounds)

        ### pass the node's inputs input_under and input_over through the node's quadratic polynomials coefficents  relu_coeffs_over and relu_coeffs_under
        num_terms_input_under = input_under.size(0) // self.n_vars
        # print(input_under)
        # print(input_under.size(0))
        num_terms_input_over = input_over.size(0) // self.n_vars
        ### degree_tensor = torch.tensor([degree, degree])
        degree_tensor = self.degree * torch.ones(self.n_vars, dtype = torch.int32)
        relu_node_under = quad_of_poly(self.n_vars, input_under, num_terms_input_under, degree_tensor, relu_coeffs_under)
        relu_node_over = quad_of_poly(self.n_vars, input_over, num_terms_input_over, degree_tensor, relu_coeffs_over)

        return relu_node_under, relu_node_over


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

    def __init__(self, n_vars, degree, layer_weights, layer_biases, layer_size):
        super().__init__()
        self.n_vars = n_vars
        self.degree = degree
        self.layer_size = layer_size
        ### get the psoitive and negative weights from the layer_weights
        self._layer_weights_pos = torch.nn.functional.relu(layer_weights)
        self._layer_weights_neg = (-1) * torch.nn.functional.relu((-1) * layer_weights)
        self.layer_biases = layer_biases
        ### call self.node object
        self.node = NodeModule(self.n_vars, self.degree)

    def forward(self, layer_inputs_under, layer_inputs_over):
        ### for each node in the layer: propagate layer_inputs_under and layer_inputs_over through the node's weights and pass them through the node 
        ### TO DO: parallize this operation by batches of nodes
        results_under = []
        results_over = []
        # print(self.layer_size)
        for i in range(self.layer_size):
            ### get node 's lower input node_under 
            # print(layer_inputs_over.size())
            # print(torch.reshape(self._layer_weights_neg[i], (self._layer_weights_neg.size(1), 1)).size())
            combined_node_under_1 = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_over, torch.reshape(self._layer_weights_neg[i], (self._layer_weights_neg.size(1), 1)))
            combined_node_under_2 = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_under, torch.reshape(self._layer_weights_pos[i], (self._layer_weights_pos.size(1), 1)))
            node_under = torch.cat((combined_node_under_1, combined_node_under_2), 0)
            node_under = add_with_constant(self.n_vars, node_under, self.layer_biases[i])

            ### get node 's upper input node_over
            combined_node_over_1 = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_over, torch.reshape(self._layer_weights_pos[i], (self._layer_weights_pos.size(1), 1)))
            combined_node_over_2 = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_under, torch.reshape(self._layer_weights_neg[i], (self._layer_weights_neg.size(1), 1)))
            node_over = torch.cat((combined_node_over_1, combined_node_over_2), 0)
            node_over = add_with_constant(self.n_vars, node_over, self.layer_biases[i])

            ### get node 's lower and upper outputs node_output_under node_output_over
            # print(node_under)
            # print(node_over)
            node_output_under, node_output_over =  self.node(node_under, node_over)

           

            ### append the results in results_under and results_over
            results_under.append(node_output_under)
            results_over.append(node_output_over)


        return torch.stack(results_under).reshape(-1, node_output_under.size(1)), torch.stack(results_over).reshape(-1, node_output_over.size(1))



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
        ### compute degrees = torch.tensor([2**0, 2**1, 2**2, ..., 2** (len(network_size)) - 2]): degree for the inputs for every layer
        degrees = torch.pow(2, torch.arange(len(network_size) - 1))
        # print(degrees)
        self.layers = [LayerModule(n_vars, degrees[i], network_weights[i], network_biases[i], network_size[i + 1]) for i in range(len(network_size) - 1) ]


        ### propagate the inputs_over and inputs_under through each layer in the network
    def forward(self, inputs):
        inputs_over = inputs
        inputs_under = inputs
        for layer in self.layers:
            print(inputs_under.size())
            inputs_under, inputs_over = layer(inputs_under, inputs_over)
        return inputs_under, inputs_over




### testing the NodeModule, LayerModule, and NetworkModule classes

if __name__ == "__main__":
    # # Set the default tensor type to be 32-bit float on the GPU
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


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

    n_vars = 4
    intervals = torch.tensor([[1., 2.], [3., 4.], [7., 9.], [5., 6.]])
    inputs = generate_inputs(n_vars, intervals, device = 'cuda')
    network_size = [4, 3, 3, 1]
    network_weights = []
    network_biases = []

    for i in range(len(network_size) - 1):
        weights = torch.randn(network_size[i  + 1], network_size[i])
        biases = torch.randn(network_size[i  + 1], 1)
        network_weights.append(weights)
        network_biases.append(biases)

    # network_weights = torch.tensor(network_weights)
    # network_biases =  torch.tensor(network_biases)   
    network = NetworkModule(n_vars, network_weights, network_biases, network_size)
    res_under, res_over = network(inputs)
    print(res_under)









