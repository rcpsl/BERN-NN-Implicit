import numpy as np
import time
import torch
import torch.nn as nn
from Bern_NN import NetworkModule, bern_coeffs_inputs
import matplotlib.pyplot as plt
import time
import cProfile

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# torch.set_default_tensor_type('torch.cuda.FloatTensor')



class True_SIA_Crown_BERN_NN():

    def __init__(self, dims, intervals, sizes, order, linear_iter_numb):

        self.dims = dims
        self.intervals = intervals
        self.sizes = sizes
        self.order = order
        self.linear_iter_numb = linear_iter_numb


    def bounds(self, torch_weights, torch_biases, ablation = False):
        # add seed 
        

        dims = self.dims
        intervals = self.intervals
        sizes = self.sizes
        order = self.order
        linear_iter_numb = self.linear_iter_numb

        torch_sizes = torch.tensor(sizes)
        torch_intervals = torch.tensor(intervals, dtype = torch.float32)

        network_module = NetworkModule(dims, torch_intervals, torch_weights, torch_biases, torch_sizes, order, linear_iter_numb)
            

        inputs = bern_coeffs_inputs(dims, intervals)
        inputs = torch.stack(inputs)



        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.device(0):
                res_under, res_over, network_nodes_des = network_module(inputs)
        bern_nn_time = time.time() - start_time
        # print( 'our_tool_time', bern_nn_time)
        #print('network_nodes_des', network_nodes_des)
        # ber_l_u = [[torch.min(res_under[0]).cpu(), torch.max(res_over[0]).cpu()]]
        # ber_l_u = [[torch.min(res_under[0]).cpu(), torch.max(res_over[0]).cpu()],  [torch.min(res_under[1]).cpu(), torch.max(res_over[1]).cpu()]]
        # ber_l_u = []
        # for i in range(sizes[len(sizes) - 1]):
        #     ber_bounds = [torch.min(res_under[i]).cpu(), torch.max(res_over[i]).cpu()]
        #     # ber_bounds = [torch.min(res[i]).cpu(), torch.max(res[i]).cpu()]
        #     ber_l_u.append(ber_bounds)
        # ber_l_u  = np.array(ber_l_u) 
        # print('our_tool_bounds:')
        # print(ber_l_u)



        return res_under, res_over, bern_nn_time




if __name__ == "__main__":

    
    dims = 2
    # intervals = np.array([[-1, 1] for _ in range(dims)])
    # intervals = np.array([[-1, 1], [2, 3]])
    # intervals = np.array([[-1.0, 1.0], [2.0, 3.0]])
    intervals = np.array([[-0.8, 0.9] , [-0.5, 0.6]])
    NN_sizes = [dims, 20, 20, 1]
    ReLU_order = 2
    linear_iter_numb = 0

    tools = True_SIA_Crown_BERN_NN(dims, intervals, NN_sizes, ReLU_order, linear_iter_numb)

    true_bounds, sia_bounds, crown_bounds, Bern_NN_bounds = tools.bounds()

    print('true_bounds\n', true_bounds)
    print('sia_bounds\n',  sia_bounds)
    print('crown_bounds\n', crown_bounds)
    print('Bern_NN_bounds\n', Bern_NN_bounds)


