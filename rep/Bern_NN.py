# disable all tensorflow debug
import os
import sys
import numpy as np
import torch
import argparse
#from relu_coeffs import relu_monom_coeffs
from relu_coeffs_pytorch import relu_monom_coeffs
from polyMultiply import bern_multiply
import time
import pickle

def nCr( n, r ):
    """
    Binomial Coefficients of form n choose r 

    n: an integer
    r: an integer less than n
    """
    return torch.exp( (torch.lgamma( torch.tensor( [n+1, n-r+1, r+1] ) ) * torch.tensor( [1, -1, -1] )).sum() ).item()

def nCr_list( N, R ):
    """
    Binomial Coefficients of form n choose r 

    N: list of integers
    R: list of integers
    """
    a = torch.lgamma(N+1)
    b = torch.lgamma(N-R+1)
    c = torch.lgamma(R+1)
    return torch.exp(torch.sum(a-b-c, 0))
    #return torch.exp(torch.sum(torch.lgamma(torch.hstack([N+1, N-R+1, R+1])) * torch.tensor([1, -1, -1], dtype=torch.float32), 1) )


def multi_dim_nCr(n_list, r_list):
    n_list = n_list.reshape(-1, 1)
    r_list = r_list.reshape(-1, 1)

    return torch.prod(nCr_list(n_list, r_list))  


def multi_dim_binom(degree):
    if(isinstance(degree, list)):
        degree = torch.hstack(degree)
    dims = len(degree)
    sizes = torch.Size((degree + 1).long())
    res = torch.ones(sizes, dtype=torch.float32)   
    indices = torch.nonzero(res)
    N = degree
    R = indices
    a = torch.lgamma(N+1)
    b = torch.lgamma(N-R+1)
    c = torch.lgamma(R+1)
    res = torch.exp(torch.sum(a-b-c, 1)).reshape(sizes)
    
    """
    for ind in indices:
        res[tuple(ind)] = multi_dim_nCr(degree, ind)
    """
    return res    

def prod_input_weights(dims, inputs, weights):
    weighted = []
    for i in range(len(weights)):
        num_nonzeros_input_i = len(torch.nonzero(torch.abs(inputs[i])))
        if (num_nonzeros_input_i != 0) and (torch.abs(weights[i][0]) != 0): 
            weighted_input = inputs[i] * weights[i][0]
            weighted.append(weighted_input)

    if len(weighted) == 0:
        weighted =   torch.zeros(dims*[1])
        return weighted      

  
    combined = torch.zeros(dims*[1])
    for i in range(len(weighted)):
        degree1 = torch.tensor(combined.shape)
        degree1 = degree1 - 1
        degree2 = torch.tensor(weighted[i].shape)
        degree2 = degree2 - 1

        sum_module = SumModule(dims, degree1, degree2)

        combined = sum_module(combined, weighted[i])
    

    return combined

### linear_combination of network inputs and weights
def linear_combination(dims, network_inputs, weights):
    sum = torch.zeros(dims*[1])
    for i in range(len(network_inputs)):
        sum = sum + network_inputs[i] * weights[i][0]
    return sum


# ========================================================
#   Compute relu of x
# ========================================================
def relu(x):
    return (abs(x) + x) / 2
    
    
def relu_bern_coefficients(interval, order):
    
    coeffs = []
    
    for k in range(order + 1):
        
        c = relu((interval[1] - interval[0]) * (k / order) + interval[0])
        coeffs.append(c)
        
        
    return coeffs      



def layer_desc(layer_indx, layer_size, layer_bounds):

    res = {}
    nodes = []
    for i in range(layer_size):
        
        node_dict = {}
        node_dict['node_indx'] = i
        node_dict['min_max'] = layer_bounds[i]

        if layer_bounds[i][0] >= 0:

            node_dict['des'] = 0

        elif layer_bounds[i][1] <= 0:

            node_dict['des'] = 1

        else:

            node_dict['des'] = 2

        nodes.append(node_dict)  

    res['layer_indx'] =  layer_indx
    res['nodes'] =  nodes  
    # with open(f"layer_dicts/weights_{layer_indx}.pkl",'wb') as f:
    #     pickle.dump(res,f)
    return res   









def control_points_matrix(degree, intervals):
    sizes = torch.Size(degree + 1)
    mat = torch.ones(sizes, dtype=torch.float32)   

    indices = torch.nonzero(mat)
    intervals_diff = intervals[:, 1] - intervals[:, 0]
    lowers = intervals[:, 0]
    res = indices * intervals_diff / degree + lowers
    ones = torch.ones(res.shape[0]).reshape(-1, 1)
    return torch.hstack([res, ones]), indices



def lower_affine_bern_coeffs(dims, network_inputs, intervals, bern_coeffs, degree):

    # intervals = torch.tensor(intervals)
    intervals_diff = intervals[:, 1] - intervals[:, 0]
    lowers = intervals[:, 0]

    A, indices = control_points_matrix(degree, intervals)
    bern_coeffs = bern_coeffs.reshape( [-1, 1])

    AT = A.T
    AT_A = torch.matmul(AT, A)
    # print(AT.shape)
    # print(bern_coeffs.shape)
    AT_b = torch.matmul(AT, bern_coeffs)


    gamma = torch.linalg.lstsq(AT_A, AT_b, rcond=None)[0]
    # print('gamma', gamma)

    # compute maximum error delta
    indices = (intervals_diff * indices) / (torch.tensor(degree)) + lowers
    cvals = torch.matmul(indices, gamma[:-1]) + gamma[-1]
    errors = bern_coeffs - cvals 
    delta = errors.max()
    gamma[-1] = gamma[-1] - delta
 
    # print('network_input', network_inputs.shape)
    # print('gamma.shape', gamma.shape)
    # print('gammagammaebfebfebf', gamma)
    lower_affine_bern_coefficients = linear_combination(dims, network_inputs, gamma[:-1]) + gamma[-1]


    return lower_affine_bern_coefficients, delta



def upper_affine_bern_coeffs(dims, network_inputs, intervals, bern_coeffs, degree): 

    minus_bern_coeffs = (-1) * bern_coeffs
    lower_affine_bern_coefficients, delta = lower_affine_bern_coeffs(dims, network_inputs, intervals, minus_bern_coeffs, degree)

    upper_affine_bern_coefficients =   (-1) * lower_affine_bern_coefficients 

    return upper_affine_bern_coefficients, delta


def cyc_intervals(listt):
    
    
    if len(listt) == 2:
        return listt
    else:
        mod_listt = listt[2:]
        mod_listt.reverse()
        res = mod_listt + listt[0:2] 
        return res


def un_cyc_intervals(listt):

    if len(listt) == 2:
        return listt
    else:
        mod_listt = listt[0:-2]
        mod_listt.reverse()
        res = listt[len(mod_listt):] + mod_listt 
        return res       




def ber_one_input(var_index, dims, interval):
    
    sizes = [2] * dims
    input_axis = torch.ones((dims, 2))
    input_axis[var_index, :] = torch.tensor(interval)
    
    input = input_axis[0, :]
    for i in range(dims - 1):

        input = torch.kron(input, input_axis[i + 1, :])

    input = input.reshape(sizes)

    return input   



def bern_coeffs_inputs(dims, intervals):
    intervals = list(intervals)
    intervals =  cyc_intervals(intervals)
    inputs = []
    for i in range(dims):

        input = ber_one_input(i, dims, intervals[i])
        inputs.append(input)


    inputs = un_cyc_intervals(inputs)
    return inputs    

def degree_elevation_poly(dims, poly, degree_poly, degree_elevated):
        
        diff_sizes = [degree_elevated[i]  - degree_poly[i] + 1 for i in range(dims)]
        multi_dim_ones_diff = torch.ones(diff_sizes)
        multi_dim_ones_diff = multi_dim_ones_diff.to('cuda')
        result = bern_multiply(multi_dim_ones_diff, poly)
        return result




class SumModule(torch.nn.Module):

    def __init__(self, dims, degree1, degree2):
        """
        power: int; power term is being raised to
        terms: int; number of xi terms
        degree: int; degree of polynomial + 1
        final_power: int, the power output should be of, used for degree elevation
        """
        super().__init__()
        self.dims   = dims
        self.degree1 = degree1
        self.degree2 = degree2



        if self.degree1[0] > self.degree2[0]:
            
            self.diff_sizes = [self.degree1[i]  - self.degree2[i] + 1 for i in range(self.dims)]
            
        else:
            self.diff_sizes = [self.degree2[i]  - self.degree1[i] + 1 for i in range(self.dims)]



        

    def forward( self, input1, input2 ):


        input1 = input1.to('cuda')
        input2 = input2.to('cuda')
        multi_dim_ones_diff = torch.ones(self.diff_sizes)
        multi_dim_ones_diff = multi_dim_ones_diff.to('cuda')

        num_nonzeros_inpu1 = len(torch.nonzero(torch.abs(input1)))
        num_nonzeros_inpu2 = len(torch.nonzero(torch.abs(input2)))



        if (num_nonzeros_inpu1 == 0) and (num_nonzeros_inpu2 == 0):
            # print('input1 == 0 and input2 == 0')
            return torch.zeros(self.dims*[1]) 

        if (num_nonzeros_inpu1 == 0) and (num_nonzeros_inpu2 != 0):
            # print('input1 == 0 and input2 != 0')
            # print(input1)
            # print(input2)
            return input2 

        if (num_nonzeros_inpu1 != 0) and (num_nonzeros_inpu2 == 0):
            # print('input1 != 0 and input2 == 0')
            return input1
            
        if (torch.sum(self.degree1)== 0) or (torch.sum(self.degree2)== 0):
            # print('degree1 == 0 or degree2 == 0')
            result = input1 + input2
            return result
        
        if torch.all(self.degree1 == self.degree2):
            # print('self.degree1 == self.degree2')
            result = input1 + input2
            return result

        else:
            if self.degree1[0] > self.degree2[0]:

                
                result = bern_multiply(multi_dim_ones_diff, input2)

                result = input1 + result 

                # print(result)

                return result
            else:

                result = bern_multiply(multi_dim_ones_diff, input1)

                
                result = input2 + result 

                return result



class PowerModule(torch.nn.Module):

    def __init__( self, dims, degree, power):
        """
        power: int; power term is being raised to
        terms: int; number of xi terms
        degree: int; degree of polynomial + 1
        final_power: int, the power output should be of, used for degree elevation
        """
        super().__init__()
        self.dims   = dims
        self.degree = degree
        self.power  = power


    def forward( self, inputs ):

        inputs = inputs.to('cuda')

        if self.power == 0:
            shp = self.dims * [1]
            power_result = torch.ones(shp)
            return power_result

        elif self.power == 1:
            power_result = inputs
            return power_result

        else:


            power_result = inputs
     
            for i in range(2, self.power + 1):

                power_result = bern_multiply(power_result, inputs)



            return power_result


class ReLUModule(torch.nn.Module):
    """
    Module that takes a polynomial in the form of a list of coefficients and then
    computes the composition of it and the incoming polynomial
    """

    def __init__( self, dims, coefficients, degree):
        """
        coefficients: list-like, list of coefficients for each term in composing polynomial, size will be degree + 1
        degree: int; degree of the input polynomial + 1
        terms: int; number of xi terms in input polynomial
        """
        super().__init__()
        
        self.dims = dims
        self.coefficients = coefficients
        self.degree = degree
        self.modules = []

        for power, coef in enumerate( self.coefficients ):
            if coef != 0:
                self.modules.append( (coef, PowerModule( self.dims, self.degree, power) ) )
        self.powerModules =  self.modules 

    def forward( self, inputs ):

        total = torch.zeros(self.dims * [1])
        if torch.any(self.coefficients):
            # shp = self.dims * [1]
            for coef, module in self.powerModules:
                if coef != 0:
                    result = module( inputs )
                    degree1 = torch.tensor(total.shape) -1 
                    #degree1 = [degree1[i] - 1 for i in range(self.dims)]
                    degree2 = torch.tensor(result.shape) - 1
                    #degree2 = [degree2[i] - 1 for i in range(self.dims)]
                    sum_module = SumModule(self.dims, degree1, degree2)
                    result = coef * result
                    total = sum_module(total, result)
        

        return total

class NodeModule(torch.nn.Module):
    """
    Module that represents a node with incoming inputs. Weights will be applied to each input and then
    summed together before being put through a relu approximation
    """
    def __init__( self, dims, coefficients_under, coefficients_over):
        """
        coefficients: list-like; list of coefficients for composing polynomial, size will be degree + 1
        degree: int, degree of the input polynomial + 1
        terms: int; number of xi terms in the input polynomial
        weights: tensor, weights for the inputs into the node
        """
        super().__init__()
        self.dims = dims
        self.coefficients_under = coefficients_under
        self.coefficients_over = coefficients_over


    def forward( self, combined_under, combined_over ):
        """
        sum_module = SumModule(self.dims, self.degree, torch.ones(self.dims), self.gpus)
        weighted_over = inputs_over * self._weights_pos.reshape([-1] + [1 for i in range(self.dims)])  + inputs_under * self._weights_neg.reshape([-1] + [1 for i in range(self.dims)]) 
        weighted_under = inputs_over * self._weights_neg.reshape([-1] + [1 for i in range(self.dims)])  + inputs_under * self._weights_pos.reshape([-1] + [1 for i in range(self.dims)]) 
        combined_over = torch.sum(weighted_over, 0)
        combined_over = sum_module(combined_over, self.bias * torch.ones(self.dims * [1]))
        combined_under = torch.sum( weighted_under, 0 )
        combined_under = sum_module(combined_under, self.bias * torch.ones(self.dims * [1]))
        """




        degree_over = torch.tensor(combined_over.shape) - 1
        # combined_over = combined_over + self.bias

        degree_under = torch.tensor(combined_under.shape) - 1
        # combined_under = combined_under + self.bias
        # s_time = time.perf_counter()
        relu_under = ReLUModule(self.dims, self.coefficients_under, degree_under)
        relu_over = ReLUModule(self.dims, self.coefficients_over, degree_over)
        # print(f"Time per node:{time.perf_counter() - s_time:.4f}")
        return relu_under(combined_under), relu_over(combined_over)


class LayerModule(torch.nn.Module):
    """
    Module that represent a fully connected layer. Is composed of multiple nodes and will return a list of 
    outputs
    """
    def __init__( self, dims, degree, weights, biases, size, order, num_layers):
        """
        coefficients: list-like; list of coefficients for composing polynomial, size will be degree + 1
        degree: int, degree of the input polynomial + 1
        terms: int; number of xi terms in the input polynomial
        weights: tensor, weights for the inputs into the node, where Xij is the weight from the ith neuron of
        the last layer to the jth neuron of this layer
        size: int; number of nodes in the layer
        """
        super().__init__()
        self.dims = dims
        self.degree = degree
        # self._weights = weights.permute(*torch.arange(weights.ndim-1, -1, -1))
        self._weights = weights
        self.biases = biases
        self._weights_pos = torch.nn.functional.relu(self._weights)
        self._weights_neg = -torch.nn.functional.relu((-1) * self._weights)
        self.size = size
        self.order = order
        self.num_layers = num_layers


    def forward( self, inputs_under, inputs_over,layer_idx):
        combined_over_list = []
        combined_under_list = []

        for i in range(self.size):
            combined_over_1 = prod_input_weights(self.dims, inputs_over, torch.unsqueeze(self._weights_pos[i, :], 1)) 
            degree_combined_over_1 = torch.tensor(combined_over_1.shape) - 1
            combined_over_2 = prod_input_weights(self.dims, inputs_under, torch.unsqueeze(self._weights_neg[i, :], 1))
            degree_combined_over_2 = torch.tensor(combined_over_2.shape) - 1
            sum_module = SumModule(self.dims, degree_combined_over_1, degree_combined_over_2)

            combined_over = sum_module(combined_over_1, combined_over_2)
            combined_over = combined_over + self.biases[i]
            combined_over_list.append(combined_over)

            combined_under_1 = prod_input_weights(self.dims, inputs_over, torch.unsqueeze(self._weights_neg[i, :], 1)) 
            degree_combined_under_1 = torch.tensor(combined_under_1.shape) - 1
            combined_under_2 = prod_input_weights(self.dims, inputs_under, torch.unsqueeze(self._weights_pos[i, :], 1))
            degree_combined_under_2 = torch.tensor(combined_under_2.shape) - 1
            sum_module = SumModule(self.dims, degree_combined_under_1, degree_combined_under_2)
            combined_under = sum_module(combined_under_1, combined_under_2)
            combined_under = combined_under + self.biases[i]
            combined_under_list.append(combined_under)
        
        
        self.bern_coefficients_over = combined_over_list
        self.bern_coefficients_under = combined_under_list

        for j in range(self.size):
            degree_over = torch.tensor(self.bern_coefficients_over[j].shape) - 1


        
        self.bounds = [[torch.min(self.bern_coefficients_under[i]), torch.max(self.bern_coefficients_over[i])] for i in range( self.size )]

        self.bounds_lower_berns = [[torch.min(self.bern_coefficients_under[i]), torch.max(self.bern_coefficients_under[i])] for i in range( self.size )]

        self.coeffs_objs = relu_monom_coeffs(self.order, self.bounds, self.bounds_lower_berns)  
        coefficients_over, coefficients_under = self.coeffs_objs.get_monom_coeffs_layer()
        self.coefficients_over = coefficients_over

        self.coefficients_under = coefficients_under
        self.nodes = [NodeModule( self.dims, self.coefficients_under[i], self.coefficients_over[i]) for i in range(self.size)]
        
        if layer_idx == self.num_layers:
            return self.bern_coefficients_under, self.bern_coefficients_over
        else:
            results_under = []
            results_over = []
            k = 0
            for node in self.nodes:
                combined_under = self.bern_coefficients_under[k]
                combined_over = self.bern_coefficients_over[k]
                res_under, res_over = node( combined_under, combined_over)
                results_under.append(res_under)
                results_over.append(res_over)
                k = k + 1

            return results_under, results_over


class NetworkModule(torch.nn.Module):
    """
    Module that represent a fully connected layer. Is composed of multiple nodes and will return a list of 
    outputs
    """
    def __init__( self, dims, intervals, layer_weights, layer_biases, sizes, order, linear_iter_numb):
        """
        coefficients: list-like; list of coefficients for composing polynomial, size will be degree + 1
        degree: int, degree of the input polynomial + 1
        terms: int; number of xi terms in the input polynomial
        weights: tensor, weights for the inputs into the node, where Xij is the weight from the ith neuron of
        the last layer to the jth neuron of this layer
        size: int; number of nodes in the layer
        """
        super().__init__()
        self.dims = dims
        self.intervals = intervals
        self.layer_weights = layer_weights
        self.layer_biases = layer_biases
        self.sizes = sizes
        self.order = order
        self.linear_iter_numb = linear_iter_numb
        self.num_layers = len(sizes) - 1
        self.network_nodes_des = []

    def forward(self, inputs):
        network_inputs = inputs
        self.inputs_over = inputs
        self.inputs_under = inputs
        

        for i in range(1, self.num_layers + 1):
            # s_time = time.perf_counter()
            # print('#################################################################################' + 'layer' + str(i) + '###################################################################')
            
            sizes = torch.tensor(self.inputs_over[0].shape)
            degree = sizes - 1
            layer_module = LayerModule( self.dims, degree, self.layer_weights[i - 1], self.layer_biases[i - 1],  self.sizes[i], self.order, self.num_layers)    
            
            self.inputs_under, self.inputs_over  = layer_module(self.inputs_under, self.inputs_over, layer_idx = i)




            if (self.linear_iter_numb != 0) and (i % self.linear_iter_numb == 0):
                
                inputs_over_linear = []
                inputs_under_linear = []
                for j in range(self.sizes[i]):
                    ################################################ over linearization ########################################################
                    degree_over = torch.tensor(self.inputs_over[j].shape) - 1
                    degree_over_sum = torch.sum(degree_over)

                    if (degree_over_sum != 0) and (degree_over_sum != self.dims):
       
                        inputs_over_jth = upper_affine_bern_coeffs(self.dims, network_inputs, self.intervals, self.inputs_over[j], degree_over)[0]
                        inputs_over_linear.append(inputs_over_jth)
                    else:
                        inputs_over_linear.append(self.inputs_over[j])

                    ################################################ under linearization ########################################################
                    degree_under = torch.tensor(self.inputs_under[j].shape) - 1
                    degree_under_sum = torch.sum(degree_under)

                    if (degree_under_sum != 0) and (degree_under_sum != self.dims):
        
                        inputs_under_jth = lower_affine_bern_coeffs(self.dims, network_inputs, self.intervals, self.inputs_under[j], degree_under)[0]
                        inputs_under_linear.append(inputs_under_jth)
                    else:
                        inputs_under_linear.append(self.inputs_under[j])    



                self.inputs_over = inputs_over_linear
                self.inputs_under = inputs_under_linear

            
            layer_bounds = [[torch.min(self.inputs_under[j]), torch.max(self.inputs_over[j])] for j in range(self.sizes[i])]

            layer_nodes_des = layer_desc(i, self.sizes[i], layer_bounds)
            self.network_nodes_des.append(layer_nodes_des)



        return self.inputs_under, self.inputs_over, self.network_nodes_des     
