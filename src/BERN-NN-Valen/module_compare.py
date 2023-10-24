import argparse
import numpy as np
import torch
from torch_modules import NetworkModule as OldNetworkModule, bern_coeffs_inputs
from torch_lists_modules import NetworkModule, generate_input, convert_to_dense
#from torch.profiler import profile, record_function, ProfilerActivity


if __name__ == "__main__":
    n_vars = 1
    #dims = 3
    intervals = [[-1, 1] for _ in range(n_vars)]
    #sizes = [dims, 10,10, 2]
    #sizes = [n_vars, 2, 1]
    #sizes = [n_vars, 2, 1]
    sizes = [n_vars, 2, 2]
    #sizes = [dims, 10, 10, 2]
    order = 2
    linear_iter_numb = 0
    gpus = 2
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--saved', action='store_true')
    args = parser.parse_args()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    layer_weights = []
    layer_biases = []
    if args.saved:
        npzfile = np.load('bad_weights.npz')
        total_weights = npzfile['weights']
        total_biases = npzfile['biases']
        for i in range(len(sizes)-1):
            layer_weights.append(total_weights[i])
            layer_biases.append(total_biases[i])
    else:
        #np.random.seed(3)
        np.random.seed(7)
        for i in range(len(sizes) - 1):
            weights = np.random.randn(sizes[i], sizes[i  + 1])
            biases = np.random.randn(sizes[i  + 1])
            layer_weights.append(weights)
            layer_biases.append(biases)

    torch_weights = []
    torch_biases = []
    for i in range(len(layer_weights)):
        torch_weights.append(torch.tensor(layer_weights[i], dtype=torch.float32))
        torch_biases.append(torch.tensor(layer_biases[i], dtype=torch.float32))

    torch_sizes = torch.tensor(sizes)
    torch_intervals = torch.tensor(intervals)
    network_module = OldNetworkModule(n_vars, torch_intervals, torch_weights, torch_biases, torch_sizes, order, linear_iter_numb, gpus)

    print("OLD FORMAT")
    inputs = bern_coeffs_inputs(n_vars, torch_intervals)
    with torch.no_grad():
        with torch.cuda.device(0):
            result_under_orig, result_over_orig, network_nodes_des = network_module(inputs)


    #print("OLD INPUTS", inputs)
    print("ORIG UNDER", result_under_orig)
    print("ORIG OVER", result_over_orig)

    """
    torch.set_default_tensor_type('torch.FloatTensor')
    for i in range(len(layer_weights)):
        torch_weights[i] = torch.tensor(layer_weights[i], dtype=torch.float32).to('cpu')
        torch_biases[i] = torch.tensor(layer_biases[i], dtype=torch.float32).to('cpu')
    """

    mod = NetworkModule(n_vars, torch_weights, torch_biases, sizes, order)
    inputs = generate_input(intervals, n_vars, device='cuda')

    print("\nNEW FORMAT")
    result_under, result_over = mod(inputs)
    dense_under = convert_to_dense(result_under, sizes[-1], n_vars, result_under_orig[0].shape[0]-1)
    dense_over = convert_to_dense(result_over, sizes[-1], n_vars, result_under_orig[0].shape[0]-1)

    #print("NEW INPUTS", inputs)
    print("NEW  UNDER", dense_under)
    print("NEW  OVER", dense_over)
    print("NEW DEVICE", dense_under.device, dense_over.device)

    i = 2
    if not torch.allclose(dense_under.cpu(), result_under_orig.to('cpu')):
        print("UNDER DOES NOT MATCH")
        print(torch.max(torch.abs(dense_under.cpu() - result_under_orig.to('cpu'))))
        i =1
    if not torch.allclose(dense_over.cpu(), result_over_orig.to('cpu')):
        print("OVER DOES NOT MATCH")
        print(torch.max(torch.abs(dense_over.cpu() - result_over_orig.to('cpu'))))
        i = 0
    if i == 2:
        print("MATCHED")
    else:
        #weights = np.stack(layer_weights)
        #biases = np.stack(layer_biases)
        #np.savez("bad_weights.npz", weights=weights, biases=biases)
        print("SAVING WEIGHTS")
