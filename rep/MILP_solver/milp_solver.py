#!/usr/bin/env python3.7

import numpy as np
import gurobipy as gp
from gurobipy import GRB, LinExpr 
import time
import torch
import pickle
import torch.nn as nn
from tools_general import True_SIA_Crown_BERN_NN
from convert_multi_bern_to_pow_poly import multi_bern_to_pow_poly

torch.set_default_tensor_type('torch.cuda.FloatTensor')



class MILP_SOLVER():

    def __init__(self, dims, intervals, sizes):

        self.dims = dims
        self.intervals = intervals
        self.sizes = sizes


    def bounds(self, weights, biases, ablation = False):


        # set parameters for NN
        # dims = 2
        # intervals = [[0, 1] for _ in range(dims)]
        # sizes = [dims, 50, 50, 1]


        # set weights and biases of NN
        layer_weights = weights
        layer_biases = biases

        # for i in range(len(sizes) - 1):
        #     weights = np.random.randn(sizes[i  + 1], sizes[i])
        #     biases = np.random.randn(sizes[i  + 1])

        #     layer_weights.append(weights)
        #     layer_biases.append(biases)



        ############################################################### solving via MILP ##################################################################### 

        # create a new model
        gmodel = gp.Model("mip")

        # make Gurobi not printing the optimization information
        gmodel.Params.LogToConsole = 0

        # create input variables
        x = gmodel.addVars(self.dims, name = "x")

        # create input constraints x
        gmodel.addConstrs((x[i] <= self.intervals[i][1] for i in range(self.dims)), name = 'c0')
        gmodel.addConstrs((x[i] >= self.intervals[i][0] for i in range(self.dims)), name = 'c1')


        # create hidden layers output variables z
        num_z_vars = sum(self.sizes) - self.dims - 1
        z = gmodel.addVars(num_z_vars, name = "z")

        # create binary variables delta
        num_delta_vars = num_z_vars 
        delta = gmodel.addVars(num_delta_vars, vtype = GRB.BINARY, name = "delta")


        # create layer constraints Ck, k = 1,..., len(sizes) - 2
        M = 15000
        index_aux = 0
        for i in range(1, len(self.sizes) - 1): # i = 1,...,len(sizes) - 2
            
            start_index = index_aux
            end_index = start_index + self.sizes[i]


        
            # constraints for every node in the layer
            for j in range(start_index, end_index):

                if i == 1:

                    weights_node = layer_weights[i - 1][j - start_index]
                    # 1 constraint
                    # print(weights_node)
                    # print(x)
                    lexpr1 = LinExpr(weights_node, x.select('*')) + layer_biases[i - 1][j - start_index] 
                    gmodel.addConstr(z[j] >= lexpr1 , name = 'l1' + str(i))

                    # 2 constraint
                    lexpr2 = LinExpr(weights_node, x.select('*')) + layer_biases[i - 1][j - start_index] +  M * delta[j]
                    gmodel.addConstr(z[j] <= lexpr2 , name = 'l2' + str(i))


                    # 4 constraint
                    gmodel.addConstr(z[j] <=  M * (1 - delta[j]), name = 'l4' + str(i))
                
                else:

                    weights_node = layer_weights[i - 1][j - start_index]
                    bias_node = layer_biases[i - 1][j - start_index]
                    # 1 constraint
                    # lexpr1 = LinExpr(weights_node, z[start_index : end_index].select('*')) + layer_biases[i - 1][j - start_index]
                    # gmodel.addConstr(z[j] >= lexpr1 , name = 'l1' + str(i))
                    gmodel.addConstr((z[j] >= sum(weights_node[k - start_index] * z[k] for k in range(start_index_before, end_index_before)) + bias_node ), name = 'l1' + str(i))

                    # 2 constraint
                    # lexpr2 = LinExpr(weights_node, z[start_index : end_index].select('*')) + layer_biases[i - 1][j - start_index] +  M * delta[j]
                    # gmodel.addConstr(z[j] <= lexpr2 , name = 'l2' + str(i))
                    gmodel.addConstr((z[j] <= sum(weights_node[k - start_index] * z[k]  for k in range(start_index_before, end_index_before)) + bias_node +  M * delta[j]  ), name = 'l2' + str(i))

                    


                    # 4 constraint
                    gmodel.addConstr(z[j] <=  M * (1 - delta[j]), name = 'l4' + str(i))
            
            if i != (end_index - 1):
                start_index_before = start_index
                end_index_before = end_index
                index_aux = end_index  






        # create NN output variable y
        y = gmodel.addVar(lb = -GRB.INFINITY, name = "y")


        # create output constraints 
        weight_output = layer_weights[-1][0]
        bias_output = layer_biases[-1][0]
        # lexpr3 = LinExpr(weight_output, z[-sizes[-2]:].select('*')) + layer_biases[-1][0]
        # gmodel.addConstrs(y == lexpr3)
        # print(start_index, end_index)
        # print(weight_output)
        # print(bias_node)
        gmodel.addConstr((y == sum(weight_output[k - start_index] * z[k]  for k in range(start_index, end_index)) + bias_output), name = 'o' )



        start_time = time.time()
        # Set objective: maximize y
        gmodel.setObjective(1.0 * y, GRB.MAXIMIZE)

        # solve the optimization problem
        gmodel.optimize()

        # get the solution
        NN_max = gmodel.objval


        # Set objective: minimize y
        gmodel.setObjective(1.0 * y, GRB.MINIMIZE)

        # solve the optimization problem
        gmodel.optimize()

        # get the solution
        NN_min = gmodel.objval

        milp_time = time.time() - start_time

        # print(NN_min, NN_max)
        # print(milp_time)

        milp_bounds = [NN_min, NN_max]

        milp_volume = NN_max - NN_min
        return milp_volume, milp_time


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



if __name__ == "__main__":
    
    # add seed 
    np.random.seed(5)
    # set parameters for NN
    dims = 4
    intervals = [[-10, 10] for _ in range(dims)]
    sizes = [dims, 50, 50, 1]
    

    # nn_file = "/home/wael/hscc23/bernn/HSCC_23_REP/experiments/table_tools_n/n_4/model1.pkl"
    # with open(nn_file, "rb") as f:
    #     model = pickle.load(f)
    # weights, biases, NN_sizes = get_weights_from_torch(model)
    # # print(weights)
    # layer_weights = []
    # layer_biases = []
    # for weight in weights:
    #     layer_weights.append(weight.cpu().detach().numpy())

    # for bias in biases:
    #     layer_biases.append(bias.cpu().detach().numpy())

    # print(layer_weights)
    # print(layer_biases)
    # print(NN_sizes)

    layer_weights = []
    layer_biases = []
    for i in range(len(sizes) - 1):
        weights = np.random.randn(sizes[i  + 1], sizes[i])
        biases = np.random.randn(sizes[i  + 1])
        # weights = np.random.uniform(size = (sizes[i  + 1], sizes[i]), low = -5, high = 5)
        # biases = np.random.uniform(size = sizes[i  + 1], low = -5, high = 5)

        layer_weights.append(weights)
        layer_biases.append(biases)

    
    # applying MILP solver
    # milp_solver = MILP_SOLVER(dims, intervals, sizes)

    # milp_volume, milp_time = milp_solver.bounds(layer_weights, layer_biases)

    # print(milp_volume)
    # print(milp_time)

    # applying BERN_NN

    torch_weights = []
    torch_biases = []
    for i in range(len(layer_weights)):
        torch_weights.append(torch.tensor(layer_weights[i], dtype=torch.float32))
        torch_biases.append(torch.tensor(layer_biases[i], dtype=torch.float32))

    ReLU_order = 4
    linear_iter_numb = 0
    tools = True_SIA_Crown_BERN_NN(dims, intervals, sizes, ReLU_order, linear_iter_numb)

    Bern_NN_lower, Bern_NN_over, bern_nn_time  = tools.bounds(torch_weights, torch_biases)

    M = torch.max(Bern_NN_over[0]).cpu()


    from_ber_to_poly = multi_bern_to_pow_poly(dims, intervals)
    bern_nn_vol = from_ber_to_poly.Bern_NN_volume(Bern_NN_lower, Bern_NN_over)

    print(bern_nn_vol)
    print(bern_nn_time)
    



    









    
    



