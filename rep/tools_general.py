import numpy as np
import time
import torch
import torch.nn as nn
from Bern_NN import NetworkModule as NetworkModuleEBF, bern_coeffs_inputs
from Bern_NN_IBF import NetworkModule as NetworkModuleIBF, generate_inputs
from Bern_NN_IBF_lin_1 import NetworkModule as NetworkModuleIBF_lin_1, generate_inputs_nn_linear
import matplotlib.pyplot as plt
import time
import cProfile
from utils import get_acas_weights
from vnnlib import VNNLib_parser
from intervals.interval_network import IntervalNetwork
from operators.linear import *
from operators.flatten import *
from operators.activations import *
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# torch.set_default_tensor_type('torch.cuda.FloatTensor')



class True_SIA_Crown_BERN_NN():

    def __init__(self, dims, intervals, sizes, order, linear_iter_numb, linear_iter_numb_ebf):

        self.dims = dims
        self.intervals = intervals
        self.sizes = sizes
        self.order = order
        self.linear_iter_numb = linear_iter_numb
        self.linear_iter_numb_ebf = linear_iter_numb_ebf

    def compute_init_bounds(self, int_net, input_bounds,method = 'symbolic'):
                TORCH_PRECISION = torch.float32
                device = 'cuda'
                # bounds_timer = time.perf_counter()
                if(method == 'symbolic'):
                    I = torch.zeros((input_bounds.shape[0], 
                                    input_bounds.shape[0]+ 1), dtype = TORCH_PRECISION)
                    I = I.fill_diagonal_(1).unsqueeze(0)#.detach()
                    input_bounds = input_bounds.unsqueeze(0).unsqueeze(1)#.detach()
                    layer_sym = SymbolicInterval(input_bounds.to(device),I,I, device = device)
                    layer_sym.conc_bounds[:,0,:,0] = input_bounds[...,0]
                    layer_sym.conc_bounds[:,0,:,1] = input_bounds[...,1]
                    # layer_sym.concretize()
                    # tic = time.perf_counter()
                    int_net(layer_sym)
                    # logger.debug(f"Initial bound propagation took {time.perf_counter() - tic:.2f} sec")
                else:
                    raise NotImplementedError()
                
                stable_relus = {}
                unstable_relus= {}
                # bounds = {}
                for l_idx, layer in enumerate(int_net.layers):
                    #Save the original bounds
                    # if type(layer) in [ReLU, Linear]:
                    #     bounds[l_idx] = {'lb': layer.post_conc_lb.squeeze(), 'ub': layer.post_conc_ub}
                    if type(layer) == ReLU:
                        stable_relus[l_idx] = []
                        unstable_relus[l_idx] = set([])
                        lb = layer.pre_conc_lb.flatten()
                        ub = layer.pre_conc_ub.flatten()
                        active_relu_idx = torch.where(lb > 0)[0]
                        inactive_relu_idx = torch.where(ub <= 0)[0]
                        unstable_relu_idx = torch.where((ub > 0) * (lb <0))[0]
                        try:
                            assert active_relu_idx.shape[0] + inactive_relu_idx.shape[0] \
                                    + unstable_relu_idx.shape[0] == lb.shape[0]
                        except AssertionError as e:
                            pass
                            # logger.error("Assertion failed(Shape mismatch): Some Relus are neither stable nor unstable")

                        stable_relus[l_idx].extend([(relu_idx.item(), 1) for relu_idx in active_relu_idx])
                        stable_relus[l_idx].extend([(relu_idx.item(), 0) for relu_idx in inactive_relu_idx])
                        unstable_relus[l_idx].update([relu_idx.item() for relu_idx in unstable_relu_idx])
                        # unstable_relus[l_idx] = unstable_relus[l_idx][::-1]
                return stable_relus, unstable_relus, layer_sym


    def bounds(self, model, torch_weights, torch_biases, ablation = False):
        # add seed 
        

        dims = self.dims
        intervals = self.intervals
        sizes = self.sizes
        order = self.order
        linear_iter_numb = self.linear_iter_numb
        linear_iter_numb_ebf = self.linear_iter_numb_ebf

        torch_sizes = torch.tensor(sizes)
        torch_intervals = torch.tensor(intervals, dtype = torch.float32)

        # model, torch_weights, torch_biases = self.get_custom_model(sizes)

        num_samples = 1000
        l, u = torch_intervals[:, 0], torch_intervals[:, 1]
        input_samples = torch.rand(num_samples,dims) * (u-l) + l
        outs = model(input_samples)
        # print(outs.cpu().detach().numpy().reshape(-1, 1))
        true_lb = outs.min(dim=0)[0].cpu().detach().numpy().reshape(-1, 1)
        true_ub = outs.max(dim=0)[0].cpu().detach().numpy().reshape(-1, 1)
        # true_lb = true_lb[0]
        # print(true_lb)
        true_bounds = np.concatenate((true_lb, true_ub), axis=1)
        # print("Min/MAX", outs.min(dim=0)[0], outs.max(dim=0)[0])
        # print('true_bounds:')
        # print(true_bounds)
        


        
        start_time = time.time() 
        COMPUTE_SIA = not ablation
        if(COMPUTE_SIA):
            device = 'cuda'
            op_dict ={"Flatten":Flatten, "ReLU": ReLU, "Linear": Linear, "Conv2d": Conv2d, "Reshape": Reshape}
            torch_intervals = torch.tensor(intervals, dtype = torch.float32).to(device)
            # parser = ONNX_Parser(path_to_model)
            # torch_model = parser.to_pytorch().to(device)
            torch_model = model
            with torch.no_grad():
                int_network = IntervalNetwork(torch_model, torch_intervals, 
                                        operators_dict=op_dict, in_shape = torch.tensor([self.dims])).to(device)
                stable_relus, unstable_relus, input_interval = self.compute_init_bounds(int_network, torch_intervals, method = 'symbolic')

            # Relu layers are @ index 2*i for i = 1,..., len(layers)/2
            sia_lb = int_network.layers[-1].post_conc_lb.cpu().numpy()
            sia_ub = int_network.layers[-1].post_conc_ub.cpu().numpy()
            sia_bounds = np.vstack((sia_lb, sia_ub)).T
            # print('sia_bounds:')
            # print(sia_bounds)
            sia_lb = int_network.layers[-1].post_symbolic.l.cpu().numpy()
            sia_ub = int_network.layers[-1].post_symbolic.u.cpu().numpy()
            # pass    
            # print('sia_bounds:')
            # print(sia_bounds)
            # unstable_neurons = 0
            # for nonlin in unstable_relus.values():
            #     unstable_neurons += len(nonlin)
            # print('unstable relus:',unstable_neurons)
            sia_bounds =  np.vstack((sia_lb, sia_ub))
            # sia_bounds = sia_bounds[:, 0]
            sia_time = time.time() - start_time
            # print( 'sia_time', sia_time)

        start_time = time.time() 
        COMPUTE_CROWN = not ablation
        if(COMPUTE_CROWN):
            from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
            device = 'cuda'
            
            # parser = ONNX_Parser(path_to_model)
            # torch_model = parser.to_pytorch().to(device)
            torch_model = model
            my_input = torch.zeros(1,sizes[0]).to(device)
            # Wrap the model with auto_LiRPA.
            model = BoundedModule(torch_model, my_input)
            # Define perturbation. Here we add Linf perturbation to input data.
            
            ptb = PerturbationLpNorm(norm=np.inf, x_L = torch_intervals[:,0].reshape(1,-1), x_U= torch_intervals[:,1].reshape(1,-1))
            # Make the input a BoundedTensor with the pre-defined perturbation.
            my_input = BoundedTensor(my_input, ptb)
            # Regular forward propagation using BoundedTensor works as usual.
            prediction = model(my_input)
            # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
            lb, ub = model.compute_bounds(x=(my_input,), method="alpha-crown")
            crown_bounds = torch.concat((lb,ub),dim=0).cpu().numpy().T
            # print('crown_bounds:')
            # print(crown_bounds)
            crown_time = time.time() - start_time
            # print( 'crown_time', crown_time)
            



        #cProfile.runctx("network_module(inputs)", globals(), locals(), "/home/simulator/pytorch_gpu")
        
        ### call network_moduleEBF and compute the results and time
        inputs = bern_coeffs_inputs(dims, intervals)
        inputs = torch.stack(inputs)
        network_moduleEBF = NetworkModuleEBF(dims, torch_intervals, torch_weights, torch_biases, torch_sizes, order, linear_iter_numb)
        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.device(0):
                res_under, res_over, network_nodes_des = network_moduleEBF(inputs)
        bern_nn_time = time.time() - start_time

        ### call network_moduleIBF and compute the results and time
        # inputs_ibf = generate_inputs(dims, torch_intervals)
        inputs_ibf= generate_inputs_nn_linear(dims)
        # network_moduleIBF = NetworkModuleIBF(dims, torch_intervals, torch_weights, torch_biases, torch_sizes, linear_iter_numb_ebf)
        network_moduleIBF = NetworkModuleIBF_lin_1(dims, torch_intervals, torch_weights, torch_biases, torch_sizes)
        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.device(0):
                res_under_ibf, res_over_ibf, res_under_degrees, res_over_degrees = network_moduleIBF(inputs_ibf)
        bern_nn_ibf_time = time.time() - start_time
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

        if (COMPUTE_SIA) and (COMPUTE_CROWN):
            # res_under = 0.0
            # res_over = 0.0
            # bern_nn_time = 0.0
            return model, true_bounds, sia_bounds, crown_bounds, res_under, res_over, res_under_ibf, res_over_ibf, res_under_degrees, res_over_degrees, sia_time, crown_time, bern_nn_time, bern_nn_ibf_time

        else:
            # res_under = 0.0
            # res_over = 0.0
            # bern_nn_time = 0.0
            return model, true_bounds, res_under, res_over, res_under_ibf, res_over_ibf, res_under_degrees, res_over_degrees, bern_nn_time, bern_nn_ibf_time




if __name__ == "__main__":

    
    dims = 2
    # intervals = np.array([[-1, 1] for _ in range(dims)])
    # intervals = np.array([[-1, 1], [2, 3]])
    # intervals = np.array([[-1.0, 1.0], [2.0, 3.0]])
    intervals = np.array([[-0.8, 0.9] , [-0.5, 0.6]])
    NN_sizes = [dims, 20, 20, 1]
    ReLU_order = 2
    linear_iter_numb = 0

    tools = True_SIA_Crown_BERN_NN(dims, intervals, NN_sizes, ReLU_order, linear_iter_numb, linear_iter_numb)

    true_bounds, sia_bounds, crown_bounds, Bern_NN_bounds = tools.bounds()

    print('true_bounds\n', true_bounds)
    print('sia_bounds\n',  sia_bounds)
    print('crown_bounds\n', crown_bounds)
    print('Bern_NN_bounds\n', Bern_NN_bounds)


