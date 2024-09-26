import torch
import torch.nn as nn
from poly_utils_cuda import *
from relu_coeffs import *
from lower_upper_linear_ibf import *
import time

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:516"

import ibf_minmax_cpp



def generate_linear_ibf(n_vars, intervals, coeffs, device = 'cuda'):
    # print('coeffs', coeffs)
    coeffs = coeffs.reshape(-1)
    inputs = []
    for i in range(n_vars + 1):

        if i != n_vars:
            ith_input = torch.ones((n_vars, 2), dtype = torch.float32, device = device)
            ith_input[i, :] = intervals[i]
            ith_input[0, :] *=  coeffs[i] 

            inputs.append(ith_input)

        else:
            ith_input = torch.ones((n_vars, 2), dtype = torch.float32, device = device)
            # print('coeffs[i]', coeffs[i])
            ith_input[0, :] *= coeffs[i] 
            inputs.append(ith_input)
    
    ### stack all inputs
    inputs = torch.cat(inputs, dim = 0)
    # print('inputs', inputs)
    return inputs


###################################################
### create a 2D inputs of the input layer for the network
###################################################
def generate_inputs_nn_linear(n_vars, device = 'cuda'):
    ### inputs = torch.eye(n_vars, dtype = torch.float32, device = device) and add a zero column to inputs
    inputs = torch.eye(n_vars, dtype = torch.float32, device = device)
    inputs = torch.cat((inputs, torch.zeros(n_vars, 1, dtype = torch.float32, device = device)), dim = 1)
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
def ibf_tensor_prod_input_weights(n_vars, inputs, weights):
    # print('weightsweightsweightsweightsweightsweights', weights)
    # print('inputinputinputinputinputinputinputinputinputinputin', inputs)
    # print('inputs_degreesinputs_degreesinputs_degreesinputs_degreesinputs_degrees', inputs_degrees)
    ### if all the weights are zeros then node_input = torch.zeros(n_vars, 1) and degree_node_input = torch.zeros(n_vars)
    ### multiply the row weights with the inputs
    if torch.all(torch.abs(weights) == 0.0) or torch.all(torch.abs(inputs) == 0.0):
        res = torch.zeros(n_vars + 1)
        degree = torch.zeros(n_vars)
        return res, degree
    
    weights = weights.reshape(1, -1)
    res = torch.matmul(weights, inputs)
    degree = torch.ones(n_vars)
    
    return res, degree






###################################################
### class NodeModule() takes as inputs:
### input_under and input_over and their degree
### and pass them through he node quadratic bounds
### q_under(input_under); q_over(input_over)
###################################################
class NodeModule(torch.nn.Module):

    def __init__(self, n_vars, intervals, node_pos_weights, node_neg_weights, node_bias, activation):
        super(NodeModule, self).__init__()
        self.n_vars = n_vars
        self.intervals = intervals
        self.node_pos_weights = node_pos_weights
        self.node_neg_weights = node_neg_weights
        self.node_bias = node_bias
        self.activation = activation


    def forward(self, layer_inputs_under, layer_inputs_over, layer_inputs_under_degrees, layer_inputs_over_degrees):

            
        ### for each node in the layer: propagate layer_inputs_under and layer_inputs_over through the node's weights and pass them through the node
        ### get node 's lower input node_under 
        # print('layer_inputs_under ', layer_inputs_under)
        # print('layer_inputs_over ', layer_inputs_over)
        # print('layer_inputs_over_degrees ', layer_inputs_over_degrees)
        combined_node_under_1, combined_node_under_1_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_over, self.node_neg_weights)
        # print('kabakabakabakabakabakabakabakabakabakabakabakabakabakabakabakabakabakabakabakabakabakaba')
        combined_node_under_2, combined_node_under_2_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_under, self.node_pos_weights)

        ### sum the two tensors combined_node_under_1 and combined_node_under_2
        input_under = combined_node_under_1 + combined_node_under_2
        ### get input_under's degree
        # print('combined_node_under_1_degree ', combined_node_under_1_degree)
        # print('combined_node_under_2_degree ', combined_node_under_2_degree)
        input_under_degree = torch.max(combined_node_under_1_degree, combined_node_under_2_degree)
        # print('input_under_degree ', input_under_degree)
        ## delete combined_node_under_1 and combined_node_under_1_degree
        del combined_node_under_1
        del combined_node_under_1_degree
        ## delete combined_node_under_2 and combined_node_under_2_degree
        del combined_node_under_2
        del combined_node_under_2_degree
        ### add the bias to input_under
        input_under[-1] = input_under[-1] + self.node_bias

        ### get node 's upper input input_over
        combined_node_over_1, combined_node_over_1_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_over, self.node_pos_weights)
        combined_node_over_2, combined_node_over_2_degree = ibf_tensor_prod_input_weights(self.n_vars, layer_inputs_under, self.node_neg_weights)
        # print('combined_node_over_1 ', combined_node_over_1)
        # print('combined_node_over_2 ', combined_node_over_2)
        ### sum the two tensors combined_node_over_1 and combined_node_over_2
        input_over = combined_node_over_1 + combined_node_over_2
        # print('input_over ', input_over)
        ### get input_over's degree
        # print('combined_node_over_1_degree ', combined_node_over_1_degree)
        # print('combined_node_over_2_degree ', combined_node_over_2_degree)
        input_over_degree = torch.max(combined_node_over_1_degree, combined_node_over_2_degree)
        # print('input_over_degree ', input_over_degree)
        ## delete combined_node_over_1 and combined_node_over_1_degree
        del combined_node_over_1
        del combined_node_over_1_degree
        ## delete combined_node_over_2 and combined_node_over_2_degree
        del combined_node_over_2
        del combined_node_over_2_degree
        ### add the bias to node_over
        input_over[-1] = input_over[-1] + self.node_bias

        # print('input_under ', input_under)
        # print('input_over ', input_over)

        # print('activation ', self.activation)
        
        ### convert input_under and input_over to ibf tensors
        input_under_coeffs = input_under
        input_over_coeffs = input_over
        input_under = generate_linear_ibf(self.n_vars, self.intervals, input_under_coeffs)
        input_over = generate_linear_ibf(self.n_vars, self.intervals, input_over_coeffs)


        if self.activation == 'linear':
            return input_under, input_over, input_under_degree, input_over_degree

        if self.activation == 'relu':
            
            input_under = torch.reshape(input_under, (input_under.size(0) // self.n_vars, self.n_vars, input_under.size(1)))
            input_over = torch.reshape(input_over, (input_over.size(0) // self.n_vars, self.n_vars, input_over.size(1)))
            ### get the bounds for the node's relu:  l = min(input_under) and u = max(input_over)
            # print(input_under_clone)
            l = ibf_minmax_cpp.ibf_minmax(input_under)[0] 
            u = ibf_minmax_cpp.ibf_minmax(input_over)[1] 
            input_under = torch.reshape(input_under, (input_under.size(0) * self.n_vars, input_under.size(2)))
            input_over = torch.reshape(input_over, (input_over.size(0) * self.n_vars, input_over.size(2)))
            
            if l > u:
                temp = l
                l = u
                u = temp
            # print('l, u ', l, u)
            

            
            
            ### if u <= 0: return relu_node_under = relu_node_over = torch.tensor([0.]), and their degrees = torch.zeros(self.n_vars)
            if u <= 0.0:
                return torch.zeros(1, self.n_vars + 1), torch.zeros(1, self.n_vars + 1), torch.zeros(self.n_vars), torch.zeros(self.n_vars)
            ### if l >= 0: return relu_node_under = input_under and relu_node_over = input_over and their degrees = input_under_degree and input_over_degree
            elif l >= 0.0:
                return input_under_coeffs, input_over_coeffs, input_under_degree, input_over_degree
            
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
                    # relu_node_under = torch.tensor([0.])
                    # relu_node_under = torch.zeros(self.n_vars, 1)
                    ### now keep the same next steps for relu_node_over
                    ### pass the node's inputs input_over through the node's quadratic polynomials coefficents  relu_coeffs_over 
                    # print(input_under)
                    # print(input_under.size(0))
                    num_terms_input_over = input_over.size(0) // self.n_vars
                    relu_node_over = quad_of_poly(self.n_vars, input_over, num_terms_input_over, input_over_degree, relu_coeffs_over)
                    ### compute the degree of relu_node_over = 2.0 * input_over_degree
                    input_over_degree = 2.0 * input_over_degree
                    return torch.zeros(1, self.n_vars + 1), relu_node_over, torch.zeros(self.n_vars), input_over_degree
                
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
                    return input_under_coeffs, relu_node_over, input_under_degree, input_over_degree
                    
                else:
                    ### pass the node's inputs input_under and input_over through the node's quadratic polynomials coefficents  relu_coeffs_over and relu_coeffs_under
                    num_terms_input_under = input_under.size(0) // self.n_vars
                    num_terms_input_over = input_over.size(0) // self.n_vars
                    # print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
                    # print('num_terms_input_under ', num_terms_input_under)
                    # print('input_under.shape ', input_under.shape)
                    # print('11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
                    relu_node_under = quad_of_poly(self.n_vars, input_under, num_terms_input_under, input_under_degree, relu_coeffs_under)
                    # print('relu_node_under.shape ', relu_node_under.shape)
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

    def __init__(self, n_vars, intervals, layer_weights, layer_biases, layer_size, activation):
        super().__init__()
        self.n_vars = n_vars
        self.intervals = intervals
        self.layer_size = layer_size
        ### get the psoitive and negative weights from the layer_weights
        self._layer_weights_pos = torch.nn.functional.relu(layer_weights)
        # print('self._layer_weights_pos ', self._layer_weights_pos)
        self._layer_weights_neg = (-1) * torch.nn.functional.relu((-1) * layer_weights)
        # print('self._layer_weights_neg ', self._layer_weights_neg)
        self.layer_biases = layer_biases
        self.activation = activation
        ### call self.nodes objects
        # print('self._layer_weights_pos ', self._layer_weights_pos)
        self.nodes = [NodeModule(self.n_vars, self.intervals, self._layer_weights_pos[i, :], self._layer_weights_neg[i, :], self.layer_biases[i], self.activation) for i in range(self.layer_size)]


    def forward(self, layer_inputs_under, layer_inputs_over, layer_inputs_under_degrees, layer_inputs_over_degrees):
        # print('layer_inputs_under ', layer_inputs_under)
        # print('layer_inputs_over ', layer_inputs_over)

        ### for each node in the layer: propagate layer_inputs_under and layer_inputs_over through the node's weights and pass them through the node 
        ### TO DO: parallize this operation by batches of nodes
        results_under = []
        results_over = []
        results_under_degrees = []
        results_over_degrees = []
        # print('layer_size ', self.layer_size)
        for node in self.nodes:
            # print('########################## node number ', i, ' ##########################')
            node_output_under, node_output_over, node_output_under_degree, node_output_over_degree = node(layer_inputs_under, layer_inputs_over, layer_inputs_under_degrees, layer_inputs_over_degrees)
            # print('node_output_under.shape ', node_output_under.shape)
           
            # ### get the min and max of node_output_under and node_output_over
            # input_under = node_output_under
            # input_over = node_output_over
            # input_under = torch.reshape(input_under, (input_under.size(0) // self.n_vars, self.n_vars, input_under.size(1)))
            # input_over = torch.reshape(input_over, (input_over.size(0) // self.n_vars, self.n_vars, input_over.size(1)))
            # ### get the bounds for the node's relu:  l = min(input_under) and u = max(input_over)
            # # print(input_under_clone)
            # # l = ibf_minmax_cpp.ibf_minmax(input_under)[0]
            # # u = ibf_minmax_cpp.ibf_minmax(input_over)[1]
            # # input_under = torch.reshape(input_under, (input_under.size(0) * self.n_vars, input_under.size(2)))
            # # input_over = torch.reshape(input_over, (input_over.size(0) * self.n_vars, input_over.size(2)))
            # # # print('l_after, u_after ', l, u)
            
            
            
            
            
            
            
            
            results_under.append(node_output_under)
            results_over.append(node_output_over)
            results_under_degrees.append(node_output_under_degree)
            results_over_degrees.append(node_output_over_degree)
            del node_output_under
            del node_output_over
            del node_output_under_degree
            del node_output_over_degree
        
        del layer_inputs_over
        del layer_inputs_over_degrees
        del layer_inputs_under
        del layer_inputs_under_degrees
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

    def __init__(self, n_vars, intervals, network_weights, network_biases, network_size, delta_error):
        super().__init__()
        self.n_vars = n_vars
        self.intervals = intervals
        self.network_size = network_size
        self.delta_error = delta_error
        ### initialize the layers with the weights and biases and their sizes and the activation function
        self.layers = [LayerModule(n_vars, self.intervals, network_weights[i], network_biases[i], network_size[i + 1], 'relu') if i != len(network_size) - 2 else LayerModule(n_vars, self.intervals, network_weights[i], network_biases[i], network_size[i + 1], 'linear') for i in range(len(network_size) - 1) ]



    ### propagate the inputs_over and inputs_under through each layer in the network
    def forward(self, inputs):
        inputs_under = inputs
        inputs_over = inputs

        inputs_under_degrees = torch.eye(self.n_vars)
        inputs_over_degrees = torch.eye(self.n_vars)

        i = 0
        # print('self.layers ', len(self.layers))
        for layer in self.layers:
            # print('################################################# layer number ', i, ' #################################################')
            # print('inputs_under_degrees ', inputs_under_degrees)
            # print('inputs_over_degrees ', inputs_over_degrees)

            inputs_under, inputs_over, inputs_under_degrees, inputs_over_degrees = layer(inputs_under, inputs_over, inputs_under_degrees, inputs_over_degrees)

            ##### check if doing linearization for the layer
            if (i != len(self.network_size) - 2):
                # print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii', i)
                inputs_under_linear = []
                inputs_over_linear = []
                inputs_under_degrees_linear = []
                inputs_over_degrees_linear = []
                ### iterate through the nodes in the layer
                for j in range(layer.layer_size):

                    ################################################ over linearization ########################################################
                    degree_over = inputs_over_degrees[j]
                    degree_over_sum = torch.sum(degree_over)
                    # print('jjjjjjj', j)
                    if (degree_over_sum != 0) and (degree_over_sum != self.n_vars):
                        # print('inputs over shape before linearization ', inputs_over[j].shape)
                        input_over_linear_jth, delta = upper_linear_ibf_1_lin(self.n_vars, self.intervals, inputs_over[j], degree_over, self.delta_error)
                        # print('inputs over shape after linearization ', input_over_linear_jth.shape)
                        inputs_over_linear.append(input_over_linear_jth.reshape(1, -1))
                        inputs_over_degrees_linear_jth = torch.ones(self.n_vars)
                        inputs_over_degrees_linear.append(inputs_over_degrees_linear_jth)

                    else:
                        inputs_over_linear.append(inputs_over[j])
                        inputs_over_degrees_linear.append(inputs_over_degrees[j])


                    ################################################ under linearization ########################################################
                    # print('kkkkkkkk')
                    degree_under = inputs_under_degrees[j]
                    degree_under_sum = torch.sum(degree_under)
                    if (degree_under_sum != 0) and (degree_under_sum != self.n_vars):
                        input_under_linear_jth, delta = lower_linear_ibf_1_lin(self.n_vars, self.intervals, inputs_under[j], degree_under, self.delta_error)
                        # print('input_under_linear_jth shape ', input_under_linear_jth.shape)
                        inputs_under_linear.append(input_under_linear_jth.reshape(1, -1))
                        inputs_under_degrees_linear_jth = torch.ones(self.n_vars)
                        inputs_under_degrees_linear.append(inputs_under_degrees_linear_jth)

                        

                    else:
                        inputs_under_linear.append(inputs_under[j])
                        inputs_under_degrees_linear.append(inputs_under_degrees[j])


                # print('inputs_under_linear[0] shape ', inputs_under_linear[0].shape)
                # print('inputs_under_linear ', inputs_under_linear)
                inputs_under = torch.cat(inputs_under_linear, dim = 0)
                # print('inputs_over_linear ', inputs_over_linear)
                inputs_over = torch.cat(inputs_over_linear, dim = 0)    
                # print('inputs_over', inputs_over) 
                inputs_under_degrees = torch.stack(inputs_under_degrees_linear)
                inputs_over_degrees = torch.stack(inputs_over_degrees_linear)





            i += 1
        return inputs_under, inputs_over, inputs_under_degrees, inputs_over_degrees




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

    ### 4) tes LayerModule() class

    # n_vars = 4
    # intervals = torch.tensor([[1., 2.], [3., 4.], [7., 9.], [5., 6.]])
    # inputs = generate_inputs(n_vars, intervals, device = 'cuda')
    # layer_size = 5
    # layer_weights = torch.randn(layer_size, n_vars)
    # layer_biases = torch.randn(layer_size, 1)
    # print('inputs ', inputs)
    # input_under = input_over = inputs
    # input_under_degrees = input_over_degrees = torch.ones(n_vars, n_vars)
    # layer = LayerModule(n_vars, layer_weights, layer_biases, layer_size, 'relu')
    # layer_res = layer(input_under, input_over, input_under_degrees, input_over_degrees)
    # print(layer_res)
    

    # ## 5) test NetworkModule() class

    # n_vars = 5
    # intervals = [[-1, 1] for i in range(n_vars)]
    # intervals = torch.tensor(intervals, dtype = torch.float32)
    # inputs = generate_inputs_nn_linear(n_vars, device = 'cuda')
    # # network_size = [n_vars, 50, 50, 50, 50, 50, 50, 5]
    # network_size = [n_vars, 5, 5, 5, 1]
    # network_weights = []
    # network_biases = []

    # for i in range(len(network_size) - 1):
    #     # weights = torch.randn(network_size[i  + 1], network_size[i])
    #     weights = torch.ones(network_size[i  + 1], network_size[i])
    #     # biases = torch.randn(network_size[i  + 1], 1)
    #     biases = torch.zeros(network_size[i  + 1], 1)
    #     network_weights.append(weights)
    #     network_biases.append(biases)
  
    # time_start = time.time()
    # with torch.no_grad():
    #     network = NetworkModule(n_vars, intervals, network_weights, network_biases, network_size)
    #     res_under, res_over, res_under_degrees, res_over_degrees = network(inputs)
    # time_end = time.time()
    # print('time ', time_end - time_start)
















