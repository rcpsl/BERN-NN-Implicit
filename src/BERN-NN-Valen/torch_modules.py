# disable all tensorflow debug
import os
import sys
import numpy as np
import torch
import argparse
#from relu_coeffs import relu_monom_coeffs
from relu_coeffs_pytorch import relu_monom_coeffs
from polyMultiply import poly_multiplication


binom_cache = dict()


def bern_multiply(polyA, polyB):
    scaling_A = multi_dim_binom(torch.tensor(polyA.shape)-1)
    scaling_B = multi_dim_binom(torch.tensor(polyB.shape)-1)
    scaled_A = scaling_A * polyA
    scaled_B = scaling_B * polyB
    result = poly_multiplication(scaled_A, scaled_B)
    unscaling = multi_dim_binom(torch.tensor(result.shape)-1)
    return result / unscaling

def bern_power(poly, power):
    scaling = multi_dim_binom(torch.tensor(poly.shape)-1)
    scaled = scaling * poly
    result = scaled
    for i in range(power - 1):
        result = poly_multiplication(result, scaled)
    #result = poly_multiplication(scaled_A, scaled_B)
    unscaling = multi_dim_binom(torch.tensor(result.shape)-1)
    return result / unscaling

def multi_dim_binom(degree):
    degree_str = str(degree)
    if degree_str in binom_cache:
        return binom_cache[degree_str]
    sizes = torch.Size((degree + 1).long())
    res = torch.ones(sizes, dtype=torch.float32)   
    indices = torch.nonzero(res)
    N = degree
    R = indices
    a = torch.lgamma(N+1)
    b = torch.lgamma(N-R+1)
    c = torch.lgamma(R+1)
    res = torch.exp(torch.sum(a-b-c, 1)).reshape(sizes)
    binom_cache[degree_str] = res
    return res    

def degree_elevation(x, final_degree):
    """degree = size -1"""
    result = multi_dim_binom(torch.tensor(x.shape)-1) * x
    multi_dim_nCr_diff = multi_dim_binom(final_degree - torch.tensor(x.shape) + 1)
    result = poly_multiplication(result, multi_dim_nCr_diff)
    result = result * (1 / multi_dim_binom(final_degree))
    return result

def prod_input_weights(dims, inputs, weights):
    weights = weights.view([-1] + ([1] * (len(inputs.shape)-1)))
    weighted = inputs * weights
    return torch.sum(weighted, 0)

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
    return res   


def control_points_matrix(dims, degree, intervals):
    sizes = torch.Size(degree + 1)
    mat = torch.ones(sizes, dtype=torch.float32)   

    indices = torch.nonzero(mat)
    intervals_diff = intervals[:, 1] - intervals[:, 0]
    lowers = intervals[:, 0]
    res = indices * intervals_diff / degree + lowers
    ones = torch.ones(res.shape[0]).reshape(-1, 1)
    return torch.hstack([res, ones]), indices


def lower_affine_bern_coeffs(dims, network_inputs, intervals, bern_coeffs, degree, AT, AT_A, indices):
    #intervals = torch.tensor(intervals)
    intervals_diff = intervals[:, 1] - intervals[:, 0]
    lowers = intervals[:, 0]

    #A, indices = control_points_matrix(dims, degree, intervals)
    bern_coeffs = bern_coeffs.view([-1, 1])
    #print("bern_coeffs", bern_coeffs.shape)
    #print("AT", AT.shape)

    #AT = A.T
    #AT_A = torch.matmul(AT, A)
    AT_b = torch.matmul(AT, bern_coeffs)
    #print("AT_b", AT_b.shape)
    #print("AT_A", AT_A.shape)
    gamma = torch.linalg.lstsq(AT_A, AT_b, rcond=None)[0]
    #print("Gamma",  gamma.shape)

    # compute maximum error delta
    indices = (intervals_diff * indices) / (degree) + lowers
    cvals = torch.matmul(indices, gamma[:-1]) + gamma[-1]
    #print(gamma[:-1].shape, indices.shape)
    #input()
    errors = bern_coeffs - cvals 
    delta = errors.max()
    gamma[-1] = gamma[-1] - delta

    lower_affine_bern_coefficients = prod_input_weights(dims, network_inputs, gamma[:-1]) + gamma[-1]
    return lower_affine_bern_coefficients, delta
    return lower_affine_bern_coefficients, delta

def lower_affine_2(dims, network_inputs, intervals, bern_coeffs, degree):
    n_inputs = bern_coeffs.shape[0]
    #print("bern shape", bern_coeffs.shape)
    bern_coeffs = bern_coeffs.view(bern_coeffs.shape[0], -1).T
    #print("reshape", bern_coeffs.shape)
    intervals_diff = intervals[:, 1] - intervals[:, 0]
    lowers = intervals[:, 0]
    A, indices = control_points_matrix(dims, degree, intervals)

    AT = A.T
    AT_A = torch.matmul(AT, A)
    #print("bern_coeffs", bern_coeffs.shape)
    #print("AT", AT.shape)
    AT_b = torch.matmul(AT, bern_coeffs).T.unsqueeze(2)
    #print("AT_b", AT_b.shape)
    stacked_ATA = torch.stack([AT_A for i in range(n_inputs)]) # TODO: WHat is J
    #print("AT_A", stacked_ATA.shape)
    gamma = torch.linalg.lstsq(stacked_ATA, AT_b)[0]
    #print("Gamma",  gamma.shape)
    gamma = torch.transpose(gamma, 0, 2).view(-1, n_inputs) # TODO: JJJJJJ

    indices = (intervals_diff * indices) / (degree) + lowers
    cvals = torch.matmul(indices, gamma[:-1]) + gamma[-1]
    errors = bern_coeffs - cvals 
    delta = errors.amax(0)
    gamma[-1] = gamma[-1] - delta

    # how does prod_input_weights unroll for this???
    res = []
    for i in range(gamma.shape[1]):
        res.append(prod_input_weights(dims, network_inputs, gamma[:-1, i]) + gamma[-1, i])
    lower_affine_bern_coefficients = torch.stack(res)
    return lower_affine_bern_coefficients




def upper_affine_bern_coeffs(dims, network_inputs, intervals, bern_coeffs, degree, AT, AT_A, indices): 
    minus_bern_coeffs = (-1) * bern_coeffs
    lower_affine_bern_coefficients, delta = lower_affine_bern_coeffs(dims, network_inputs, intervals, minus_bern_coeffs, degree, AT, AT_A, indices)
    upper_affine_bern_coefficients =   (-1) * lower_affine_bern_coefficients 
    return upper_affine_bern_coefficients
    return upper_affine_bern_coefficients, delta

def upper_affine_2(dims, network_inputs, intervals, bern_coeffs, degree):
    minus_bern_coeffs = -1 * bern_coeffs
    lower_affine_bern_coefficients = lower_affine_2(dims, network_inputs, intervals, minus_bern_coeffs, degree)
    upper_affine_bern_coefficients = -1 * lower_affine_bern_coefficients
    return upper_affine_bern_coefficients




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
    """
    print(dims, intervals)
    inputs = []

    for i in range(dims):

        input_1 = ber_one_input(i, dims, intervals[i])
        inputs.append(input_1)
    print(inputs)
    input()
    return inputs    
    """
    sizes = [dims] + ([2] * dims)
    res = torch.ones(sizes)
    view_shape = [-1]
    for i in range(dims):
        res[dims-1-i] = res[dims-1-i] * intervals[dims-1-i].view(view_shape)
        view_shape.append(1)
    return res
        



class SumModule(torch.nn.Module):

    def __init__(self, dims, degree1, degree2, gpus):
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
        self.gpus   = gpus


        if degree1[0] > degree2[0]:
            self.diff_sizes = self.degree1 - self.degree2 + 1
        else:
            self.diff_sizes = self.degree2 - self.degree1 + 1

        

    def forward( self, input1, input2 ):
        #multi_dim_ones_diff = torch.ones(self.diff_sizes)

        #num_nonzeros_inpu1 = len(torch.nonzero(torch.abs(input1)))
        has_nonzero_inpu1 = torch.any(input1)
        #num_nonzeros_inpu2 = len(torch.nonzero(torch.abs(input2)))
        has_nonzero_inpu2 = torch.any(input2)

        """
        if (num_nonzeros_inpu1 == 0) and (num_nonzeros_inpu2 == 0):
            return torch.zeros(self.dims*[1]) 
        if (num_nonzeros_inpu1 == 0) and (num_nonzeros_inpu2 != 0):
            return input2 
        if (num_nonzeros_inpu1 != 0) and (num_nonzeros_inpu2 == 0):
            return input1
        """
        if not has_nonzero_inpu1 and not has_nonzero_inpu2:
            return torch.zeros(self.dims*[1])
        if not has_nonzero_inpu1 and has_nonzero_inpu2:
            return input2
        if has_nonzero_inpu1 and not has_nonzero_inpu2:
            return input1
        if (torch.sum(self.degree1)== 0) or (torch.sum(self.degree2)== 0):
            result = input1 + input2
            return result
        if torch.all(self.degree1 == self.degree2):
            result = input1 + input2
            return result
        else:
            if self.degree1[0] > self.degree2[0]:
                #result = bern_multiply(multi_dim_ones_diff, input2)
                result = degree_elevation(input2, self.degree1)
                result = input1 + result 
                return result
            else:
                #result = bern_multiply(multi_dim_ones_diff, input1)
                result = degree_elevation(input1, self.degree2)
                result = input2 + result 
                return result


class PowerModule(torch.nn.Module):

    def __init__( self, dims, degree, power, gpus):
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
        self.gpus   = gpus


    def forward( self, inputs ):
        inputs = inputs.to('cuda')
        if self.power == 0:
            shp = self.dims * [1]
            power_result = torch.ones(shp)
        elif self.power == 1:
            power_result = inputs
        else:
            """
            power_result = inputs
            for i in range(2, self.power + 1):
                power_result = bern_multiply(power_result, inputs)
            """
            power_result = bern_power(inputs, self.power)
        return power_result


class ReLUModule(torch.nn.Module):
    """
    Module that takes a polynomial in the form of a list of coefficients and then
    computes the composition of it and the incoming polynomial
    """

    def __init__( self, dims, coefficients, degree, gpus):
        """
        coefficients: list-like, list of coefficients for each term in composing polynomial, size will be degree + 1
        degree: int; degree of the input polynomial + 1
        terms: int; number of xi terms in input polynomial
        """
        super().__init__()
        
        self.dims = dims
        self.coefficients = coefficients
        self.degree = degree
        self.gpus = gpus
        self.modules = []

        for power, coef in enumerate( self.coefficients ):
            if coef != 0:
                self.modules.append( (coef, PowerModule( self.dims, self.degree, power, self.gpus) ) )
        self.powerModules = self.modules

    def forward( self, inputs ):
        final_degree = self.degree * (len(self.coefficients) - 1)
        total = 0
        if torch.any(self.coefficients):
            shp = self.dims * [1]
            for coef, module in self.powerModules:
                if coef != 0:
                    result = coef * module( inputs )
                    total += degree_elevation(result, final_degree)
        else:
            return torch.zeros(torch.Size(final_degree+1))
        return total

class NodeModule(torch.nn.Module):
    """
    Module that represents a node with incoming inputs. Weights will be applied to each input and then
    summed together before being put through a relu approximation
    """
    def __init__( self, dims, coefficients_under, coefficients_over, degree, weights, bias, gpus):
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
        self.degree = degree
        self._weights = weights
        self.bias = bias
        # TODO: check in place
        self._weights_pos = torch.nn.functional.relu(self._weights)
        self._weights_neg = -torch.nn.functional.relu((-1) * self._weights)
        self.gpus = gpus
        
        #self.relu_under = ReLUModule( self.dims, self.coefficients_under, self.degree, self.gpus)
        #self.relu_over = ReLUModule( self.dims, self.coefficients_over, self.degree, self.gpus)

    def forward( self, inputs_under, inputs_over ):
        # I don't think that inputs_under and inputs_over would ever be different degrees
        """
        combined_over_1 = prod_input_weights(self.dims, inputs_over, self._weights_pos) 
        degree_combined_over_1 = torch.tensor(combined_over_1.shape) - 1
        combined_over_2 = prod_input_weights(self.dims, inputs_under, self._weights_neg)
        degree_combined_over_2 = torch.tensor(combined_over_2.shape) - 1
        
        sum_module = SumModule(self.dims, degree_combined_over_1, degree_combined_over_2, self.gpus)
        combined_over = sum_module(combined_over_1, combined_over_2)

        combined_under_1 = prod_input_weights(self.dims, inputs_over, self._weights_neg) 
        degree_combined_under_1 = torch.tensor(combined_under_1.shape) - 1
        combined_under_2 = prod_input_weights(self.dims, inputs_under, self._weights_pos)
        degree_combined_under_2 = torch.tensor(combined_under_2.shape) - 1

        sum_module = SumModule(self.dims, degree_combined_under_1, degree_combined_under_2, self.gpus)
        combined_under = sum_module(combined_under_1, combined_under_2)
        """
        combined_over_1 = prod_input_weights(self.dims, inputs_over, self._weights_pos) 
        combined_over_2 = prod_input_weights(self.dims, inputs_under, self._weights_neg)
        combined_over = combined_over_1 + combined_over_2

        combined_under_1 = prod_input_weights(self.dims, inputs_over, self._weights_neg) 
        combined_under_2 = prod_input_weights(self.dims, inputs_under, self._weights_pos)
        combined_under = combined_under_1 + combined_under_2


        degree_over = torch.tensor(combined_over.shape) - 1
        combined_over = combined_over + self.bias

        degree_under = torch.tensor(combined_under.shape) - 1
        combined_under = combined_under + self.bias

        relu_under = ReLUModule(self.dims, self.coefficients_under, degree_under, self.gpus)
        relu_over = ReLUModule(self.dims, self.coefficients_over, degree_over, self.gpus)

        return relu_under(combined_under), relu_over(combined_over)


class LayerModule(torch.nn.Module):
    """
    Module that represent a fully connected layer. Is composed of multiple nodes and will return a list of 
    outputs
    """
    def __init__( self, dims, degree, weights, biases, size, order, gpus):
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
        self._weights = weights.permute(*torch.arange(weights.ndim-1, -1, -1))
        self.biases = biases
        self._weights_pos = torch.nn.functional.relu(self._weights)
        self._weights_neg = -torch.nn.functional.relu((-1) * self._weights)
        self.size = size
        self.order = order
        self.gpus = gpus


    def forward( self, inputs_under, inputs_over ):
        #print(inputs_under.shape, inputs_over.shape)
        # print('inputs_under', inputs_under)
        # print('inputs_over', inputs_over)
        # self.bern_coefficients_over = tuple([prod_input_weights(self.dims, inputs_over, torch.unsqueeze(self._weights_pos[i, :], 1)) + prod_input_weights(self.dims, inputs_under, torch.unsqueeze(self._weights_neg[i, :], 1)) + self.biases[i] for i in range(self.size)])
        # self.bern_coefficients_under = tuple([prod_input_weights(self.dims, inputs_over, torch.unsqueeze(self._weights_neg[i, :], 1)) + prod_input_weights(self.dims, inputs_under, torch.unsqueeze(self._weights_pos[i, :], 1))+ self.biases[i] for i in range(self.size)])
        
        combined_over_list = []
        combined_under_list = []
        for i in range(self.size):
            # I think inputs_over and inputs_under will always be the same degree
            combined_over_1 = prod_input_weights(self.dims, inputs_over, torch.unsqueeze(self._weights_pos[i, :], 1)) 
            #degree_combined_over_1 = torch.tensor(combined_over_1.shape) - 1
            combined_over_2 = prod_input_weights(self.dims, inputs_under, torch.unsqueeze(self._weights_neg[i, :], 1))
            #degree_combined_over_2 = torch.tensor(combined_over_2.shape) - 1
            #sum_module = SumModule(self.dims, degree_combined_over_1, degree_combined_over_2, self.gpus)
            #combined_over = sum_module(combined_over_1, combined_over_2)
            combined_over = combined_over_1 + combined_over_2 + self.biases[i]
            combined_over_list.append(combined_over)

            combined_under_1 = prod_input_weights(self.dims, inputs_over, torch.unsqueeze(self._weights_neg[i, :], 1)) 
            #degree_combined_under_1 = torch.tensor(combined_under_1.shape) - 1
            combined_under_2 = prod_input_weights(self.dims, inputs_under, torch.unsqueeze(self._weights_pos[i, :], 1))
            #degree_combined_under_2 = torch.tensor(combined_under_2.shape) - 1
            #sum_module = SumModule(self.dims, degree_combined_under_1, degree_combined_under_2, self.gpus)
            #combined_under = sum_module(combined_under_1, combined_under_2)
            combined_under = combined_under_1 + combined_under_2 + self.biases[i]
            combined_under_list.append(combined_under)
        
        
        self.bern_coefficients_over = combined_over_list
        self.bern_coefficients_under = combined_under_list
        #print("BERN OVER", combined_over_list)
        #print("BERN UNDER", combined_under_list)

        
        self.bounds = [[[torch.min(self.bern_coefficients_under[i]), torch.max(self.bern_coefficients_over[i])] for i in range( self.size )]]
        #print("BOUNDS", self.bounds[0], "ORDER", self.order)

        self.coeffs_objs = relu_monom_coeffs(self.order, self.bounds[0])  
        coefficients_over, coefficients_under = self.coeffs_objs.get_monom_coeffs_layer()
        self.coefficients_over = coefficients_over
        self.coefficients_under = coefficients_under
        #print("COEF OVER", self.coefficients_over)
        #print("COEF UNDER", self.coefficients_under)
        self.nodes = [NodeModule( self.dims, self.coefficients_under[i], self.coefficients_over[i], self.degree, self._weights[i], self.biases[i], self.gpus) for i in range(self.size)]
        #print("NodeModule", self._weights[0], self.biases[0])
           
        results_under = []
        results_over = []
        for node in self.nodes:
            res_under, res_over = node( inputs_under, inputs_over)
            results_under.append(res_under)
            results_over.append(res_over)
        return torch.stack(results_under), torch.stack(results_over)
        #return results_under, results_over


class NetworkModule(torch.nn.Module):
    """
    Module that represent a fully connected layer. Is composed of multiple nodes and will return a list of 
    outputs
    """
    def __init__( self, dims, intervals, layer_weights, layer_biases, sizes, order, linear_iter_numb, gpus):
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
        self.gpus = gpus  
        self.num_layers = len(sizes) - 1
        self.network_nodes_des = []

    def forward(self, inputs):
        network_inputs = inputs
        self.inputs_over = inputs
        self.inputs_under = inputs
        

        for i in range(1, self.num_layers + 1):
            print(i)
            sizes = torch.tensor(self.inputs_over[0].shape)
            degree = sizes - 1
            layer_module = LayerModule( self.dims, degree, self.layer_weights[i - 1], self.layer_biases[i - 1],  self.sizes[i], self.order, self.gpus)    
            
            self.inputs_under, self.inputs_over  = layer_module(self.inputs_under, self.inputs_over)
            #print("INPUTS", self.inputs_under.shape)
            #input()

            #layer_bounds = tuple([[torch.min(self.inputs_under[j]), torch.max(self.inputs_over[j])] for j in range(self.sizes[i])])
            a = torch.amin(self.inputs_under, tuple(range(1,len(self.inputs_under.shape)))).view(-1, 1)
            b = torch.amax(self.inputs_over, tuple(range(1,len(self.inputs_under.shape)))).view(-1, 1)
            bounds = torch.hstack([a, b])

            layer_nodes_des = layer_desc(i, self.sizes[i], bounds)

            self.network_nodes_des.append(layer_nodes_des)

            if (self.linear_iter_numb != 0) and (i % self.linear_iter_numb == 0):
                sizes = torch.tensor(self.inputs_over[0].shape)

                degree = sizes - 1

                A, indices = control_points_matrix(self.dims, degree, self.intervals)
                AT = A.T
                AT_A = torch.matmul(AT, A)
                #inputs = self.inputs_under
                #self.inputs_under = torch.stack([lower_affine_bern_coeffs(self.dims, network_inputs, self.intervals, self.inputs_under[j], degree, AT, AT_A, indices)[0] for j in range(self.sizes[i])])
                self.inputs_under = lower_affine_2(self.dims, network_inputs, self.intervals, self.inputs_under, degree)
                #self.inputs_over  = torch.stack([upper_affine_bern_coeffs(self.dims, network_inputs, self.intervals, self.inputs_over[j], degree, AT, AT_A, indices) for j in range(self.sizes[i])])
                self.inputs_over = upper_affine_2(self.dims, network_inputs, self.intervals, self.inputs_over, degree) 

        return self.inputs_under, self.inputs_over, self.network_nodes_des     


class PaddingModule(torch.nn.Module):
    def __init__( self, degree, final_degree, dims ):
        super().__init__()
        padding_amount = [lfd_i - ld_i for lfd_i, ld_i in zip(final_degree, degree)]
        self.padding_size = []
        for i in range(dims):
            self.padding_size += [0, int(padding_amount[dims-1-i])]
        self.padding_size = tuple(self.padding_size)
        

    def forward( self, inputs ):
        return torch.nn.functional.pad(inputs, self.padding_size)

if __name__ == "__main__":
    #pass



    dims = 2
    degree1 = [3, 2]
    degree1 = torch.tensor(degree1)
    degree2 = [1, 1]
    degree2 = torch.tensor(degree2)
    gpus = 7

    dims = 2
    gpus = 5

    sum_module = SumModule(dims, degree1, degree2, gpus)


    input1 = np.array([[4, 8, 16],
                       [8, 16, 32],
                       [16, 32, 64],
                       [32, 64, 128]])

    input2 = -30 * np.array([[2, 4],
                       [4, 8]])

    
    input1 = torch.tensor(input1, dtype = torch.float32)

    input2 = torch.tensor(input2, dtype = torch.float32)

    # padding = PaddingModule( degree2, degree1, dims )

    

    # print(padding(input2))

    res = sum_module(input1, input2)
    print(res)
    
