from turtle import xcor
import numpy as np
import time
import torch
import random
import argparse
from torch_modules import NetworkModule, bern_coeffs_inputs
#from dist_torch_modules import NetworkModule, bern_coeffs_inputs
import tensorflow as tf
from relu_bern_coeffs import Network
#import matplotlib.pyplot as plt
import time
import cProfile



def time_gpu():
    # add seed 
    np.random.seed(1) 
    dims = 2
    #dims = 3
    intervals = [[0, 1] for _ in range(dims)]
    #sizes = [dims, 10,10, 2]
    sizes = [dims, 1]
    #sizes = [dims, 10, 10, 2]
    order = 2
    linear_iter_numb = 1
    gpus = 2

    #weights1 = 10 * np.ones((2, 60))
    #weights2 = (1/4) * np.ones((60, 60)) - np.diag([i for i in range(60)])
    #weights3 = np.ones((60, 50)) + 0.1 
    #weights4 = np.ones((50, 2)) - 0.25

    #weights1 =  np.random.randn(sizes[0], sizes[1])
    #weights2 =  np.random.randn(sizes[1], sizes[2]) 
    #weights3 = 20 * np.random.randn(sizes[2], sizes[3]) + 30
    #weights4 = np.random.randn(sizes[3], sizes[4])
    #weights5 = np.random.randn(sizes[4], sizes[5]) + 50
    #weights6 = 20 * np.random.randn(sizes[5], sizes[6]) + 30
    #weights7 = np.random.randn(sizes[6], sizes[7])
    
    layer_weights = []
    layer_biases = []
    

    for i in range(len(sizes) - 1):
        weights = np.random.randn(sizes[i], sizes[i  + 1])
        # weights = np.ones((sizes[i], sizes[i  + 1]))
        biases = np.zeros(sizes[i  + 1])
        #biases = np.random.randn(sizes[i  + 1])
        layer_weights.append(weights)
        layer_biases.append(biases)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    torch_weights = []
    torch_biases = []
    for i in range(len(layer_weights)):
        torch_weights.append(torch.tensor(layer_weights[i], dtype=torch.float32))
        torch_biases.append(torch.tensor(layer_biases[i], dtype=torch.float32))


    torch_sizes = torch.tensor(sizes)
    torch_intervals = torch.tensor(intervals)
    network_module = NetworkModule(dims, torch_intervals, torch_weights, torch_biases, torch_sizes, order, linear_iter_numb, gpus)
    
    #input1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    #input2 = np.array([[0.0, 1.0], [0.0, 1.0]])
    #input3 = np.array([[0.0, 1.0], [0.0, 1.0]])

    #inputs = [input1, input2, input3]
    inputs = bern_coeffs_inputs(dims, torch_intervals)
    #inputs = torch.stack(inputs)
    #print(inputs[0])

    #cProfile.runctx("network_module(inputs)", globals(), locals(), "/home/simulator/pytorch_gpu")

    start_time = time.time()
    with torch.no_grad():
        with torch.cuda.device(0):
            res_under, res_over, network_nodes_des = network_module(inputs)
    elapsed_time = time.time() - start_time
    #print('network_nodes_des', network_nodes_des)
    print( 'our_tool_time', elapsed_time)
    # ber_l_u = [[torch.min(res_under[0]).cpu(), torch.max(res_over[0]).cpu()]]
    #ber_l_u = [[torch.min(res_under[0]).cpu(), torch.max(res_over[0]).cpu()],  [torch.min(res_under[1]).cpu(), torch.max(res_over[1]).cpu()]]
    ber_l_u = [[torch.min(res_under[i]).cpu(), torch.max(res_over[i]).cpu()] for i in range(len(res_under))]
    exit(0)






    activation = 'relu'
    rect = np.array(intervals)
    net = Network(sizes, activation, order, 'rand', rect)

    
    
    #weights1 = weights1.T
    #weights2 = weights2.T
    #weights3 = weights3.T
    #weights4 = weights4.T
    #weights5 = weights5.T
    #weights6 = weights6.T
    #weights7 = weights7.T
    #weights = [weights1, weights2]

    for i in range(len(sizes) - 1):
        layer_weights[i] = layer_weights[i].T
        layer_biases[i] = layer_biases[i].reshape(-1, 1)

    
    #biases1 = np.zeros((sizes[1], 1))
    #biases2 = np.zeros((sizes[2], 1))
    #biases3 = np.zeros((sizes[3], 1))
    #biases4 = np.zeros((sizes[4], 1))
    #biases5 = np.zeros((sizes[5], 1))
    #biases6 = np.zeros((sizes[6], 1))
    #biases7 = np.zeros((sizes[7], 1))
    #biases = [biases1, biases2]

    #biases = []
    #for i in range(len(sizes) - 1):
    #    biase = np.zeros((sizes[i + 1], 1))
    #    biases.append(biase)


    #print('layer_weights222222222222222222222222222222222222222222222222222222222222', layer_weights[0])  

    net._W = layer_weights
    net._b = layer_biases
    
    inputs_samples = []
    for i in range(dims):
        x = np.random.uniform(intervals[i][0], intervals[i][1], size = 8000)
        x = x.reshape(-1, 1)
        inputs_samples.append(x)

    
    #print(inputs_samples)
    X = np.concatenate(inputs_samples, axis = 1)
    Y = net.forward_prop(X.T)

    

    start_time = time.time()
    net.relu_coeffs_IA()
    elapsed_time= time.time() - start_time
    print( 'IA_time', elapsed_time)
    #print(net.IA_l_u[-1])
    IA_l_u = net.IA_l_u[-1]


    
    start_time = time.time()
    net.relu_coeffs_LP()
    elapsed_time= time.time() - start_time
    print( 'LP_time', elapsed_time)
    #print(net.LP_l_u[-1])
    LP_l_u = net.LP_l_u[-1]

    true_l_u = [ [min(Y[i, :]), max(Y[i, :])] for i in range(Y.shape[0])]
    # true_l_u = [[min(Y[0, :]), max(Y[0, :])]]


    
    true_l_u = np.array(true_l_u) 
    ber_l_u  = np.array(ber_l_u) 
    IA_l_u   = np.array(IA_l_u) 
    LP_l_u   = np.array(LP_l_u)

    """
    ####### plot

    bern_rect_x = np.array([ber_l_u[0, 0], ber_l_u[0, 1], ber_l_u[0, 0], ber_l_u[0, 1]])
    bern_rect_y = np.array([ber_l_u[1, 0], ber_l_u[1, 0], ber_l_u[1, 1], ber_l_u[1, 1]])

    IA_rect_x = np.array([IA_l_u[0, 0], IA_l_u[0, 1], IA_l_u[0, 0], IA_l_u[0, 1]])
    IA_rect_y = np.array([IA_l_u[1, 0], IA_l_u[1, 0], IA_l_u[1, 1], IA_l_u[1, 1]])

    LP_rect_x = np.array([LP_l_u[0, 0], LP_l_u[0, 1], LP_l_u[0, 0], LP_l_u[0, 1]])
    LP_rect_y = np.array([LP_l_u[1, 0], LP_l_u[1, 0], LP_l_u[1, 1], LP_l_u[1, 1]])

    fig = plt.figure()
    ax = fig.add_subplot(111)


    plt.scatter(Y[0, :], Y[1, :])
    #.scatter(true_rect_x, true_rect_y)
    plt.scatter(bern_rect_x, bern_rect_y)
    plt.scatter(IA_rect_x, IA_rect_y)
    plt.scatter(LP_rect_x, LP_rect_y)

    plt.legend([r'actual output',r'our tool', r'Interval arithmetic', r'Linear programming'], prop = {'size' : 8}, loc = 'upper right')
    plt.xlabel(r'$y_1$')
    plt.ylabel(r'$y_2$')
    plt.savefig('experiment.png')
    plt.savefig('experiment.eps')
    #plt.show() 
    """


    return true_l_u, ber_l_u, IA_l_u, LP_l_u


    


    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    parser.add_argument( "-p", "--power", type=int )
    parser.add_argument( "-t", "--terms", type=int )
    parser.add_argument( "-d", "--degree", type=int )
    """
    #parser.add_argument( "-g", "--gpus", type=int )
    #parser.add_argument( "-i", "--iterations", type=int, default=20 )
    #args = parser.parse_args()

    #exec_time = time_gpu( 0, 0, 0, 10, args.gpus )

    #throughput = args.iterations / exec_time

    #print( f"{args.gpus},{exec_time/args.iterations},{throughput}" )

    true_l_u, ber_l_u, IA_l_u, LP_l_u = time_gpu()

    print('true_l_u', list(true_l_u))
    print('bern_l_u', list(ber_l_u))
    print('IA_l_u', list(IA_l_u))
    print('LP_l_u', list(LP_l_u))

    true_l_u = np.array(true_l_u)
    ber_l_u  = np.array(ber_l_u)

    #mse = (( (ber_l_u - true_l_u) ** 2 ).mean(axis = 1)).mean()
    norm1_error = (ber_l_u - true_l_u).max()
    print('norm1_error', norm1_error)



   
 




