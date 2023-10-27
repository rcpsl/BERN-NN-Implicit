#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:10:44 2022

@author: waelfatnassi
"""


import numpy as np
import os
from scipy.optimize import linprog
from relu_coeffs_pytorch import relu_monom_coeffs

class Network:

    def __init__(self, dim_list, activation, order, weight_type, rect, fix_weights=False,
                    load_weights=False, weight_root=None):
        """Constructor for neural network class
        params:
            dim_list: list of ints      - dimensions of each layer in network
            actiavtion: str             - activation function for network
                                        - 'relu' | 'sigmoid' | 'tanh'
            rect: instance of Rectangle - upper and lower bounds on input
            weight_type: str            - 'ones' | 'rand'
            fix_weights: bool           - fix random seed for weights if True
        """

        # activation function for network
        self._activation = activation
        
        # order of relu approximation
        
        self.order = order

        # dimensions of layers
        self._dim_list = dim_list
        self._num_hidden_layers = len(dim_list) - 2
        

        self._weight_type = weight_type
        self._fix_weights = fix_weights
        self._rect = rect

        if load_weights == False:
            self._create_weights()
        else:
            self._weight_root = weight_root
            self._load_weights()


    @property
    def rect(self):
        return self._rect

    @rect.setter
    def rect(self, instance):
        self._rect = instance
        self._calc_lower_and_upper_bounds()

    def _create_weights(self):
        """Create weights and biases for 1-layer neural network"""

        # c is a normalizing constant for creating weight matrices
        c = 1 / np.sqrt(self._num_hidden_layers)

        self._W, self._b = {}, {}
        
        if self._weight_type == 'rand':

            if self._fix_weights:
                np.random.seed(5)

            for i in range(self._num_hidden_layers + 1):
                self._W[i] = c * np.random.randn(self._dim_list[i+1], self._dim_list[i])
                self._b[i] = c * np.random.randn(self._dim_list[i+1], 1)

        elif self._weight_type == 'ones':

            for i in range(self._num_hidden_layers + 1):
                self._W[i] = c * np.ones(shape=(self._dim_list[i+1], self._dim_list[i]))
                self._b[i] = c * np.ones(shape=(self._dim_list[i+1], 1))

        else:
            raise ValueError('Please use weight_type = "rand" | "ones"')

    def _load_weights(self):

        self._W, self._b = {}, {}

        weights = np.load(os.path.join(self._weight_root, 'failed_weights100.npy'), allow_pickle=True)
        biases = np.load(os.path.join(self._weight_root, 'failed_biases100.npy'), allow_pickle=True)

        for i in range(self._num_hidden_layers + 1):
            self._W[i] = weights[i]
            self._b[i] = biases[i]


    def forward_prop(self, X):
        """Compute one forward pass through the network
        params:
            X: (d,n) array of floats - training instances
        returns:
            Y: (dout,n) array of floats - output from training instances"""

        # compute for each layer in network
        # TODO: vectorize?
        Y = X
        for i in range(self._num_hidden_layers + 1):
            if i == self._num_hidden_layers:
                Y = np.matmul(self._W[i], Y) + self._b[i]
            else:
                Y = np.maximum(0, np.matmul(self._W[i], Y) + self._b[i])

        # return np.matmul(self._W[1], Y) + self._b[1]
        return Y

  
                
                                
    def output_range_layer(self, weight, bias, input_range_layer, activation):
        # solving LPs
        neuron_dim = bias.shape[0]
        output_range_box = []
        for j in range(neuron_dim):
            # c: weight of the j-th dimension
            c = weight[j]
            c = c.transpose()
            b = bias[j]
            # compute the minimal input
            res_min = linprog(c, bounds=input_range_layer, options={"disp": False})
            input_j_min = res_min.fun + b
            # compute the minimal output
            if activation == 'ReLU':
                if input_j_min < 0:
                    output_j_min = list(np.array([0]))[0]
                else:
                    output_j_min = list(input_j_min)[0]
            # if activation == 'sigmoid':
            #     output_j_min = 1/(1+np.exp(-input_j_min))
            # if activation == 'tanh':
            #     output_j_min = 2/(1+np.exp(-2*input_j_min))-1
            # compute the maximal input
            res_max = linprog(
                -c,
                bounds=input_range_layer,
                options={"disp": False}
            )
            input_j_max = -res_max.fun + b
            # compute the maximal output
            if activation == 'ReLU':
                if input_j_max < 0:
                    output_j_max = list(np.array([0]))[0]
                else:
                    output_j_max = list(input_j_max)[0]
            # if activation == 'sigmoid':
            #     output_j_max = 1/(1+np.exp(-input_j_max))
            # if activation == 'tanh':
            #     output_j_max = 2/(1+np.exp(-2*input_j_max))-1
            output_range_box.append([output_j_min, output_j_max])
        return np.array(output_range_box)
    
    
    
    def relu_coeffs_IA(self):
        """Calculate lower and upper bounds on output of linear layers and
        output of nonlinear activation functions"""
        
        
        
        # lower and upper bounds for input to (nonlinear) activation function
        # for the neurons in the hidden layer
        x_lower_bounds = {0: self._rect[:, 0].reshape(-1, 1)}
        x_upper_bounds = {0: self._rect[:, 1].reshape(-1, 1)}
        
        self.IA_l_u = []
        self.IA_coeffs = []
        
        for i in range(1, self._num_hidden_layers + 2):
        
            # lower bound on input to hidden layer
            lower_max_term = np.matmul(np.maximum(self._W[i - 1], 0), x_lower_bounds[i - 1])
            lower_min_term = np.matmul(np.minimum(self._W[i - 1], 0), x_upper_bounds[i - 1])
            x_lower_bounds[i] = lower_max_term + lower_min_term + self._b[i - 1]
    
            # upper bound on input to hidden layer
            upper_min_term = np.matmul(np.minimum(self._W[i - 1], 0), x_lower_bounds[i - 1])
            upper_max_term = np.matmul(np.maximum(self._W[i - 1], 0), x_upper_bounds[i - 1])
            x_upper_bounds[i] = upper_min_term + upper_max_term + self._b[i - 1]
    
            # lower and upper bounds for output of (nonlinear) activation function
            # for the neurons in the hidden layer
            y_lower_bounds, y_upper_bounds = {}, {}
            # print(i)
            if self._activation == 'relu':
               # if i != self._num_hidden_layers + 1:
                # print('iafter', i)
                y_lower_bounds[i] = np.maximum(x_lower_bounds[i], 0)
                y_upper_bounds[i] = np.maximum(x_upper_bounds[i], 0)
                    
                l_u = np.concatenate((y_lower_bounds[i], y_upper_bounds[i]), axis = 1).tolist()
                    
                self.IA_l_u.append(l_u)
                    
                relu_coeffs = relu_monom_coeffs(self.order, l_u)
                    
                coeffs_layer = relu_coeffs.get_monom_coeffs_layer()
                    
                self.IA_coeffs.append(coeffs_layer)
                    

                
        # print(self.IA_coeffs)        
                
                
                
    def relu_coeffs_LP(self):
        """Calculate lower and upper bounds on output of linear layers and
        output of nonlinear activation functions"""

        # lower and upper bounds for input to (nonlinear) activation function
        # for the neurons in the hidden layer
        x_lower_bounds = {0: self._rect[:, 0].reshape(-1, 1)}
        x_upper_bounds = {0: self._rect[:, 1].reshape(-1, 1)}
        
        self.LP_l_u = []
        self.LP_coeffs = []
        
        for i in range(1, self._num_hidden_layers + 2):
        
            
            #if i != self._num_hidden_layers + 1:
                # lower and upper bounds for output of (nonlinear) activation function
                # for the neurons in the hidden layer
     
            weight = self._W[i - 1]
            bias = self._b[i - 1]
            input_range_layer = np.concatenate((x_lower_bounds[i - 1].reshape(-1, 1), x_upper_bounds[i - 1].reshape(-1, 1)), axis = 1)
            activation = 'ReLU'
            l_u = self.output_range_layer(weight, bias, input_range_layer, activation)
                
                
                
            x_lower_bounds[i] = l_u[:, 0].reshape(-1, 1)
            x_upper_bounds[i] = l_u[:, 1].reshape(-1, 1)
                
            l_u = l_u.tolist()
            self.LP_l_u.append(l_u)
                
                
            relu_coeffs = relu_monom_coeffs(self.order, l_u)
                    
            coeffs_layer = relu_coeffs.get_monom_coeffs_layer()
                    
            self.LP_coeffs.append(coeffs_layer)
            
            
        # print(self.LP_coeffs)    
            

    
    def total_num_hidden_units(self):
        """Getter method for total number of hidden units in network"""

        return sum(self._dim_list[1:len(self._dim_list)-1])

    @property
    def dim_input(self):
        """Getter method for input dimension of neural network"""

        return self._dim_list[0]

    @property
    def dim_output(self):
        """Getter method for output dimension of neural network"""

        return self._dim_list[-1]

    @property
    def dim_hidden_layer(self):
        """Getter method for list of dimensions of hidden layers in neural network"""

        return self._dim_list[1]

    

    
    def IA_l_u(self):

        return self.IA_l_u
    
    
    def LP_l_u(self):

        return self.LP_l_u
    
    
    def IA_coeffs(self):

        return self.IA_coeffs
    
    
    def LP_coeffs(self):

        return self.LP_coeffs


    def _weights_as_list(self):

        return [w for w in self._W.values()]

    def _biases_as_list(self):

        return [b for b in self._b.values()]


# if __name__ == '__main__':
    
#     dim_list = [2, 3, 3, 2]
#     activation = 'relu'
#     rect = np.array([[-3, 3], [-3, 3]])
#     net = Network(dim_list, activation, 3, 'rand', rect)
    
    
   
#     print('Weights', net._W)
#     print('####################################################################')
#     print('biases', net._b)
    
    
#     net.relu_coeffs_IA()
#     print(net.IA_coeffs)
#     print('###############################')
#     net.relu_coeffs_LP()
#     print(net.LP_coeffs)
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
