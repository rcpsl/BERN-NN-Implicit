import numpy as np
from tools_general import True_SIA_Crown_BERN_NN
from convert_multi_bern_to_pow_poly import multi_bern_to_pow_poly
import argparse
import torch
import torch.nn as nn
import pickle
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# np.random.seed(1) 
# torch.manual_seed(24313425)
import pandas as pd
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn



def parse_polar_network(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        assert len(lines) >= 3, "Corrupted file"

        # Read input size
        layers_sizes = [int(lines[0])] 
        # Read hidden layers sizes
        num_hidden = int(lines[2])
        for l_idx in range(3,3 + num_hidden):
            layers_sizes.append(int(lines[l_idx]))
        # Read output size
        layers_sizes.append(int(lines[1]))
        
        # Read activations per layer
        start_idx = 3 + num_hidden # Start of reading activations
        activations =[act.strip() for act in lines[start_idx:start_idx + num_hidden + 1]]
        start_idx  = start_idx + num_hidden + 1

        # Read weights and biases
        weights_biases = [float(line) for line in lines[start_idx:]]

        prev_layer_size = layers_sizes[0]
        read_idx = 0
        weights = [] 
        biases = []
        for layer_idx, layer_size  in enumerate(layers_sizes):
            if(layer_idx == 0): continue
            num_weights = prev_layer_size * layer_size
            W = weights_biases[read_idx: read_idx + num_weights]
            W = torch.tensor(W).reshape((layer_size, prev_layer_size))
            W[torch.abs(W) < 1E-15] = 0.0
            read_idx += num_weights
            b = weights_biases[read_idx: read_idx + layer_size]
            b = torch.tensor(b)
            b[torch.abs(b) < 1E-15] = 0.0
            read_idx += layer_size
            prev_layer_size = layer_size

            weights.append(W)
            biases.append(b)

        assert (len(weights_biases) - read_idx) ==2, "Parsing error"

        return weights, biases, layers_sizes, activations


def get_weights_from_torch(model):
    weights = []
    biases = []
    NN_sizes = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                weights.append(module.weight.clone())
                biases.append(module.bias.clone())
                NN_sizes.append(module.in_features)
        
        # Assume last layer is linear!!
        NN_sizes.append(module.out_features)

    return weights, biases, NN_sizes

def get_custom_model_random(model_index, layers_sizes=[]):
        layers = []
        in_size = layers_sizes[0]
        for l_idx, l_size in enumerate(layers_sizes[1:]):
            layers.append(nn.Linear(in_size, l_size))
            if l_idx != (len(layers_sizes)-2):
                layers.append(nn.ReLU())
            in_size = l_size

        model = nn.Sequential(*layers)

        weights = []
        biases = []
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    l_w, u_w = -5, 5
                    l_b, u_b = -5, 5
                    module.weight[:] = torch.rand(*module.weight.shape, dtype=torch.float32) * (u_w - l_w) + l_w
                    module.bias[:] = torch.rand(*module.bias.shape, dtype=torch.float32) * (u_b - l_b) + l_b
                    # module.bias[:] = torch.zeros(*module.bias.shape)
                    weights.append(module.weight.clone())
                    biases.append(module.bias.clone())

############ save the model       
        # Define the directory name
        dir_name = f"experiments/fig_time_vs_n/n_{layers_sizes[0]}"   
        # Check if the directory exists, if not create it
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)  

        # Define the file path
        file_path = os.path.join(dir_name, f"model{model_index}.pkl")
        # Save the model
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)

        return model, weights, biases 

def get_custom_model_polar(layers_sizes=[]):
        layers = []
        in_size = layers_sizes[0]
        for l_idx, l_size in enumerate(layers_sizes[1:]):
            layers.append(nn.Linear(in_size, l_size))
            if l_idx != (len(layers_sizes)-2):
                layers.append(nn.ReLU())
            in_size = l_size

        model = nn.Sequential(*layers)

        weights = []
        biases = []
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    l_w, u_w = -5, 5
                    l_b, u_b = -5, 5
                    module.weight[:] = torch.rand(*module.weight.shape, dtype=torch.float32) * (u_w - l_w) + l_w
                    module.bias[:] = torch.rand(*module.bias.shape, dtype=torch.float32) * (u_b - l_b) + l_b
                    # module.bias[:] = torch.zeros(*module.bias.shape)
                    weights.append(module.weight.clone())
                    biases.append(module.bias.clone())

        return model, weights, biases 


def set_model_weights(model, weights, biases):

    index = 0
    with torch.no_grad():
        for module in model.modules():
            if(hasattr(module, "weight")):
                module.weight[:] = weights[index]
                module.bias[:] = biases[index]
                index += 1
    
    assert index == len(weights), "Model has different number of layers than the provided weights"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("relu_order", type = int)
    parser.add_argument("linear_param", type = int)
    parser.add_argument("linear_param_ibf", type = int)
    parser.add_argument("--mode", type = str, default='random')
    parser.add_argument("--nn_file", type = str)
    parser.add_argument("--dims", type = int)
    parser.add_argument("--intervals", required=True, nargs="+", type = float)
    parser.add_argument("--out_file", default ="results.csv")
    parser.add_argument("--ablation", action=argparse.BooleanOptionalAction)
    parser.add_argument("--acasxu_model", type = str)
    parser.add_argument("--acasxu_spec", type = str)

    args = parser.parse_args()

    mode = args.mode.lower()

    # ### generate 50 random models
    # if(mode == "random"):
    #     dims = args.dims
    #     NN_sizes = [dims, 20, 20, 1]
    #     for i in range(1, 51):
    #         print(i)
    #         model, weights, biases  = get_custom_model_random(i, NN_sizes)

    if(mode == "polar"):
        assert len(args.nn_file) !=0, "You must provide the path to NN file with argument. Example '--nn_file file_path'"
        fname = args.nn_file
        polar_weights, polar_biases, NN_sizes, activations = parse_polar_network(fname)
        dims = NN_sizes[0]
        model, weights, biases  = get_custom_model_polar(NN_sizes)    
        weights = polar_weights
        biases = polar_biases
        set_model_weights(model, weights, biases)

    # Bern tool config
    ReLU_order = args.relu_order                                                                                                                                 
    linear_iter_numb = args.linear_param
    linear_iter_numb_ibf = args.linear_param_ibf
    dims = args.dims
    intervals = args.intervals    
    assert len(intervals) == dims * 2, f"Interval argument should have {2 * dims} numbers"
    intervals = np.array(intervals).reshape((-1,2))
    
    if (mode == "random"):
        sia_vol_list = []
        crown_vol_list = []
        bern_nn_vol_list = []
        bern_nn_ibf_vol_list = []

        sia_time_list = []
        crown_time_list = []
        bern_nn_time_list = []
        bern_nn_ibf_time_list = []
        # for i in range(3):
        np.random.seed(1) 
        torch.manual_seed(24313435)

        # print(i)

        # Read PyTorch NN
        #nn_file = args.nn_file
        #with open(nn_file, "rb") as f:
        #    model = pickle.load(f)
        #weights, biases, NN_sizes = get_weights_from_torch(model)

        
        model, weights, biases  = get_custom_model_random(0, NN_sizes)
        set_model_weights(model, saved_weights, saved_biases)
        print(biases)
        # print('######################################################################')

        tools = True_SIA_Crown_BERN_NN(dims, intervals, NN_sizes, ReLU_order, linear_iter_numb, linear_iter_numb_ibf)

        from_ber_to_poly = multi_bern_to_pow_poly(dims, intervals)
        if(not args.ablation):
            model, true_bounds, sia_bounds, crown_bounds, Bern_NN_lower, Bern_NN_over, Bern_NN_lower_IBF, Bern_NN_over_IBF, res_under_degrees, res_over_degrees, sia_time, crown_time, bern_nn_time, bern_nn_ibf_time  = tools.bounds(model, weights, biases)
            sia_vol = from_ber_to_poly.sia_volume(sia_bounds)
            crown_vol = from_ber_to_poly.crown_volume(crown_bounds)
            # print('sia_bounds\n',  sia_bounds)
            # print('crown_bounds\n', crown_bounds)
        else:
            model, true_bounds, Bern_NN_lower, Bern_NN_over, Bern_NN_lower_IBF, Bern_NN_over_IBF, res_under_degrees, res_over_degrees, bern_nn_time, bern_nn_ibf_time  = tools.bounds(model, weights, biases, ablation = True)


        # sia_vol = from_ber_to_poly.sia_volume(sia_bounds)
        # crown_vol = from_ber_to_poly.crown_volume(crown_bounds)
        # bern_nn_vol = from_ber_to_poly.Bern_NN_volume(Bern_NN_lower, Bern_NN_over)
        bern_nn_ibf_vol = from_ber_to_poly.Bern_NN_IBF_volume(Bern_NN_lower_IBF, Bern_NN_over_IBF, res_under_degrees, res_over_degrees)
        # bern_nn_vol = torch.tensor([0])
        

        # print('sia_relative_vol\n',  float(sia_vol))
        # print('crown_relative_vol\n', float(crown_vol))
        # print('bern_nn_relative_vol\n', float(bern_nn_vol))

        # sia_vol_list.append(sia_vol)
        # crown_vol_list.append(crown_vol)
        bern_nn_vol_list.append(0.0)
        bern_nn_ibf_vol_list.append(bern_nn_ibf_vol)

        # sia_time_list.append(sia_time)
        # crown_time_list.append(crown_time)
        bern_nn_time_list.append(bern_nn_time)
        bern_nn_ibf_time_list.append(bern_nn_ibf_time)




        # print('sia_time_min', min(sia_time_list))
        # print('sia_time__max', max(sia_time_list))
        # print('sia_time_mean', sum(sia_time_list))
        # print(sia_time_list)

        # print('crown_time_min', min(crown_time_list))
        # print('crown_time_max', max(crown_time_list))
        # print('crown_time_mean', sum(crown_time_list))
        # print(crown_time_list)
        # print('bern_time_min', min(bern_nn_time_list))
        # print('bern_nn_time_max', max(bern_nn_time_list))
        # print('sia_time', sia_time)
        # print('crown_time', crown_time)
        print('bern_nn_time', bern_nn_time)
        print('bern_nn_ibf_time', bern_nn_ibf_time)
        print('######################################################################')
        true_vol = from_ber_to_poly.crown_volume(true_bounds)
        print('true_vol', true_vol)
        # print('sia_vol', sia_vol)
        # print('crown_vol', crown_vol)
        # print('bern_nn_vol', bern_nn_vol)
        print('bern_nn_ibf_vol', bern_nn_ibf_vol)
        # print('bern_nn_relative_vol_mean', sum(bern_nn_vol_list) / 10)


        if not args.ablation:
            result = {'sia_time':sia_time, 'sia_vol': sia_vol.item(),
                    'crown_time':crown_time, 'crown_vol': crown_vol.item(),
                    'bern_time':bern_nn_time, 'bern_vol': 0.0,
                    'bern_ibf_time':bern_nn_ibf_time, 'bern_ibf_vol': bern_nn_ibf_vol.item()}
        else:
            result = {'bern_time':bern_nn_time, 'bern_vol': 0.0, 'bern_ibf_time':bern_nn_ibf_time, 'bern_ibf_vol': bern_nn_ibf_vol.item()}

        out_fname = args.out_file 
        if os.path.exists(out_fname):
            df = pd.read_csv(out_fname)
            df = df._append(result, ignore_index = True)
        else:
            df = pd.DataFrame(result, [0])
        pass        

        df.to_csv(out_fname, index=False)
        



    if (mode == "polar"):


            # model, weights, biases  = get_custom_model_polar(NN_sizes)

            # set_model_weights(model, weights, biases)

            tools = True_SIA_Crown_BERN_NN(dims, intervals, NN_sizes, ReLU_order, linear_iter_numb)
            try:   
                model, true_bounds, sia_bounds, crown_bounds, Bern_NN_lower, Bern_NN_over, Bern_NN_IBF, lower, Bern_NN_IBF_over, res_under_degrees, res_over_degrees, sia_time, crown_time, bern_nn_time, bern_nn_ibf_time  = tools.bounds(model, weights, biases)

            except Exception as e: 
                print('CROWN error computing bounds, this instance will be skipped')
                sys.exit()


            from_ber_to_poly = multi_bern_to_pow_poly(dims, intervals)

            # print('true_bounds\n', true_bounds)
            # print('sia_bounds\n',  sia_bounds)
            # # print('lower bounds', sia_bounds[0])
            # # print('upper bounds', sia_bounds[1])
            # # sia_diff = sia_bounds[1] - sia_bounds[0]
            # # print('difference', sia_diff)
            # # print('coeffs', sia_diff[0][:-1])
            # # print('const', sia_diff[0][-1])
            # print('crown_bounds\n', crown_bounds)
            # print('Bern_NN_bounds\n', Bern_NN_bounds)

            sia_vol = from_ber_to_poly.sia_volume(sia_bounds)
            crown_vol = from_ber_to_poly.crown_volume(crown_bounds)
            bern_nn_vol = from_ber_to_poly.Bern_NN_volume(Bern_NN_lower, Bern_NN_over)
            bern_nn_ibf_vol = from_ber_to_poly.Bern_NN_IBF_volume(Bern_NN_IBF, Bern_NN_IBF_over, res_under_degrees, res_over_degrees)

            
            print('sia_time\n',  sia_time)
            print('crown_time\n', crown_time)
            print('bern_nn_time\n', bern_nn_time)
            print('bern_nn_ibf_time\n', bern_nn_ibf_time)

            print('sia_relative_vol\n',  float(sia_vol))
            print('crown_relative_vol\n', float(crown_vol))
            print('bern_nn_relative_vol\n', float(bern_nn_vol))
            print('bern_nn_ibf_relative_vol\n', float(bern_nn_ibf_vol))

            result = {'sia_time':sia_time, 'sia_vol': sia_vol.item(),
                    'crown_time':crown_time, 'crown_vol': crown_vol.item(),
                    'bern_time':bern_nn_time, 'bern_vol': bern_nn_vol.item(),
                    'bern_ibf_time':bern_nn_ibf_time, 'bern_ibf_vol': bern_nn_ibf_vol.item()}

            out_fname = args.out_file 
            if os.path.exists(out_fname):
                df = pd.read_csv(out_fname)
                df = df._append(result, ignore_index = True)
            else:
                df = pd.DataFrame(result, [0])
            pass        

            df.to_csv(out_fname, index=False)


    # if (mode == "acasxu"):
    #     ### read the acasxu model
    #     acasxu_nn_model = args.acasxu_model
    #     acasxu_nn_spec = args.acasxu_spec
    #     weights, biases = 






