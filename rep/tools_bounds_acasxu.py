import numpy as np
import time
import argparse
import json
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
import os
import pandas as pd

import sys
sys.path.append('/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/bern_cuda_ext')
import ibf_minmax_cpp




torch.set_default_tensor_type('torch.cuda.FloatTensor')




class Tools_Bounds():
    def __init__(self, dims, intervals, torch_weights, torch_biases, sizes, linear_iter_numb_ibf):

        self.dims = dims
        self.torch_intervals = torch.tensor(intervals, dtype = torch.float32).to('cuda')
        self.torch_weights = torch_weights
        self.torch_biases = torch_biases
        self.torch_sizes = sizes
        self.linear_iter_numb_ibf = linear_iter_numb_ibf


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
    
    def true_bounds(self, model):

        # model, torch_weights, torch_biases = self.get_custom_model(sizes)

        num_samples = 10000
        l, u = self.torch_intervals[:, 0], self.torch_intervals[:, 1]
        input_samples = torch.rand(num_samples, self.dims) * (u-l) + l
        outs = model(input_samples)
        # print(outs.cpu().detach().numpy().reshape(-1, 1))
        true_lb = outs.min(dim=0)[0].cpu().detach().numpy().reshape(-1, 1)
        true_ub = outs.max(dim=0)[0].cpu().detach().numpy().reshape(-1, 1)
        # true_lb = true_lb[0]
        # print(true_lb)
        true_bounds = np.concatenate((true_lb, true_ub), axis=1)
        return true_bounds
    
    def sia_bounds(self, model):
            start_time = time.time() 
            device = 'cuda'
            op_dict ={"Flatten":Flatten, "ReLU": ReLU, "Linear": Linear, "Conv2d": Conv2d, "Reshape": Reshape}
            # parser = ONNX_Parser(path_to_model)
            # torch_model = parser.to_pytorch().to(device)
            torch_model = model
            with torch.no_grad():
                int_network = IntervalNetwork(torch_model, self.torch_intervals, 
                                        operators_dict=op_dict, in_shape = torch.tensor([self.dims])).to(device)
                stable_relus, unstable_relus, input_interval = self.compute_init_bounds(int_network, self.torch_intervals, method = 'symbolic')

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
            # sia_bounds =  np.vstack((sia_lb, sia_ub))
            # sia_bounds = sia_bounds[:, 0]
            sia_time = time.time() - start_time

            return sia_bounds, sia_time
    

    def crown_bounds(self, model):
        start_time = time.time() 
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
        device = 'cuda'
        
        # parser = ONNX_Parser(path_to_model)
        # torch_model = parser.to_pytorch().to(device)
        torch_model = model
        my_input = torch.zeros(1,self.torch_sizes[0]).to(device)
        # Wrap the model with auto_LiRPA.
        model = BoundedModule(torch_model, my_input)
        # Define perturbation. Here we add Linf perturbation to input data.
        
        ptb = PerturbationLpNorm(norm=np.inf, x_L = self.torch_intervals[:,0].reshape(1,-1), x_U= self.torch_intervals[:,1].reshape(1,-1))
        # Make the input a BoundedTensor with the pre-defined perturbation.
        my_input = BoundedTensor(my_input, ptb)
        # # Regular forward propagation using BoundedTensor works as usual.
        # prediction = model(my_input)
        # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
        lb, ub = model.compute_bounds(x=(my_input,), method="alpha-crown")
        crown_bounds = torch.concat((lb,ub),dim=0).cpu().numpy().T
        # print('crown_bounds:')
        # print(crown_bounds)
        crown_time = time.time() - start_time
        return crown_bounds, crown_time
    

    
    def bern_ibf_bounds(self, spec):
        # inputs_ibf = generate_inputs(self.dims, self.torch_intervals)
        inputs_ibf = generate_inputs_nn_linear(self.dims)
        # print('self.torch_weights[0].shape:', self.torch_weights[0].shape)
        # print('self.torch_biases:', self.torch_biases)
        self.torch_weights = [self.torch_weights[i].T for i in range(len(self.torch_weights))]
        # network_moduleIBF = NetworkModuleIBF(self.dims, self.torch_intervals, self.torch_weights, self.torch_biases, self.torch_sizes, self.linear_iter_numb_ibf)
        if spec == "7":
            delta_error = 1.16
            network_moduleIBF = NetworkModuleIBF_lin_1(self.dims, self.torch_intervals, self.torch_weights, self.torch_biases, self.torch_sizes, delta_error)
        else:
            delta_error = 0.84
            network_moduleIBF = NetworkModuleIBF_lin_1(self.dims, self.torch_intervals, self.torch_weights, self.torch_biases, self.torch_sizes, delta_error)
        start_time = time.time()
        with torch.no_grad():
            with torch.cuda.device(0):
                res_under_ibf, res_over_ibf, res_under_degrees, res_over_degrees = network_moduleIBF(inputs_ibf)
        bern_nn_ibf_time = time.time() - start_time

        return res_under_ibf, res_over_ibf, bern_nn_ibf_time
    



def main():
    start_time = time.time()
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev", type=str)
    parser.add_argument("--tau", type=str)
    parser.add_argument("--spec", type=str)
    parser.add_argument("--out_file", default ="results.csv")
    args = parser.parse_args()

    dims = 5
    torch_sizes = torch.tensor([5, 50, 50, 50, 50, 50, 50, 5], dtype = torch.int32).to('cuda')
    linear_iter_numb_ibf = 1
    onnx_file = "/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/hscc23/bernn/HSCC_23_REP/experiments/acasxu/onnx/ACASXU_run2a_{prev}_{tau}_batch_2000.onnx".format(prev=args.prev, tau=args.tau)
    spec_file = "/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/hscc23/bernn/HSCC_23_REP/experiments/acasxu/vnnlib/prop_{spec}.vnnlib".format(spec=args.spec)
    model, torch_weights, torch_biases = get_acas_weights(onnx_file)
    ### move model, torch_weights, torch_biases to cuda
    model = model.to('cuda')
    torch_weights = [torch_weights[i].to('cuda') for i in range(len(torch_weights))]
    torch_biases = [torch_biases[i].to('cuda') for i in range(len(torch_biases))]
    # print(model)
    # print(torch_weights)
    # print(torch_biases)
    input_space = VNNLib_parser(spec_file).read_vnnlib_simple(spec_file, 5, 5)[0][0]
    print('input space: \n', np.array(input_space))

    # Create an instance of Tools_Bounds
    tool_bounds_obj = Tools_Bounds(dims, input_space, torch_weights, torch_biases, torch_sizes, linear_iter_numb_ibf)
    
    #### true bounds ####
    true_bounds = tool_bounds_obj.true_bounds(model)
    print('true bounds: \n', true_bounds)



    #### sia bounds ####
    sia_bounds, sia_time = tool_bounds_obj.sia_bounds(model)
    print('sia bounds: \n', sia_bounds)
    print('sia time: \n', sia_time)


    ### crown bounds ###
    # print('crowncrowncrowncrowncrowncrowncrowncrowncrowncrown')
    crown_bounds, crown_time = tool_bounds_obj.crown_bounds(model)
    print('crown bounds: \n', crown_bounds)
    print('crown time: \n', crown_time)


    ### bern_nn_ibf bounds ###
    # print('bernibfbernibfbernibfbernibfbernibfbernibfbernibfbernibfbernibfbernibf')
    bern_ibf_under, bern_ibf_over, bern_ibf_time = tool_bounds_obj.bern_ibf_bounds(args.spec)
    # print('kkkkkkkkkkkkkkkkkkkkkkkkkkk')
    bounds = []
    for i in range(5):
        lb = bern_ibf_under[i]
        ub = bern_ibf_over[i]
        lb =  torch.reshape(lb, (lb.size(0) // dims, dims, lb.size(1)))
        ub =  torch.reshape(ub, (ub.size(0) // dims, dims, ub.size(1)))
        lb_min, lb_max = ibf_minmax_cpp.ibf_minmax(lb)
        ub_min, ub_max = ibf_minmax_cpp.ibf_minmax(ub)
        ### convert to a value not a tensor
        lb_min = lb_min.item() 
        lb_max = lb_max.item()
        ub_min = ub_min.item()
        ub_max = ub_max.item()

        # bounds.append([lb_min, ub_max])
        bounds.append([min(lb_min, ub_max), max(lb_max, ub_min)])

    
    print('bern_nn_ibf bounds: \n', np.array(bounds))
    print('bern_nn_ibf time: \n', bern_ibf_time)


    #### save the results time and bounds into a dictionary ####
    results_dict = {}
    results_dict['true_bounds'] = [[true_bounds[i][0], true_bounds[i][1]] for i in range(true_bounds.shape[0])]
    results_dict['sia_bounds'] = [[sia_bounds[i][0], sia_bounds[i][1]] for i in range(sia_bounds.shape[0])]
    results_dict['crown_bounds'] = [[crown_bounds[i][0], crown_bounds[i][1]] for i in range(crown_bounds.shape[0])]
    results_dict['bern_ibf_bounds'] = bounds
    results_dict['sia_time'] = sia_time
    results_dict['crown_time'] = crown_time
    results_dict['bern_ibf_time'] = bern_ibf_time

    out_fname = args.out_file 
    if os.path.exists(out_fname):
        df = pd.read_csv(out_fname)
        df = df._append(results_dict, ignore_index = True)
    else:
        df = pd.DataFrame([results_dict])
    pass        

    df.to_csv(out_fname, index=False)


    




if __name__ == "__main__":
    main()



