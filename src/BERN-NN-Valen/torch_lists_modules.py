import numpy as np
import torch

from copy import deepcopy

from relu_coeffs_pytorch import relu_monom_coeffs
from relu_lists import relu, find_range, combine, degree_elevation_list, bern_scaling_list, bern_unscaling_list


from torch_modules import LayerModule as OldLayerModule, degree_elevation


def prod_input_weights(inputs, weights):
    """
    weights = weights.view([-1] + ([1] * (len(inputs.shape)-1)))
    weighted = inputs * weights
    return torch.sum(weighted, 0)
    """
    result = []
    for ten, weight in zip(inputs, weights):
        if weight != 0:
            for t in ten:
                term = t+ 0
                term[0,:] *= weight
                result.append(term)
    return result


class NodeModule(torch.nn.Module):

    def __init__(self, n_vars, order, weights, bias):
        super().__init__()
        self.n_vars = n_vars
        self.order = order
        self._weights = weights

        # Store bias in new format
        self.bias = torch.ones((n_vars, 1))
        self.bias[0,0] = bias

        # TODO: check in place for neg maybe
        self._weights_pos = torch.nn.functional.relu(self._weights)
        self._weights_neg = -torch.nn.functional.relu((-1) * self._weights)

    def forward(self, inputs_under, inputs_over):
        combined_over_1 = prod_input_weights(inputs_over, self._weights_pos) 
        combined_over_2 = prod_input_weights(inputs_under, self._weights_neg)
        bern_coefficients_over = combined_over_1 + combined_over_2

        combined_under_1 = prod_input_weights(inputs_over, self._weights_neg) 
        combined_under_2 = prod_input_weights(inputs_under, self._weights_pos)
        bern_coefficients_under = combined_under_1 + combined_under_2
        
        bern_coefficients_over = bern_coefficients_over + [self.bias]
        bern_coefficients_under = bern_coefficients_under + [self.bias]

        #print("BERN OVER", bern_coefficients_over)
        #print("BERN UNDER", bern_coefficients_under)
        #print("BERN OVER", combine(bern_coefficients_over, self.n_vars))
        #print("BERN UNDER", combine(bern_coefficients_under, self.n_vars))

        _, bounds_min = find_range(bern_coefficients_under, self.n_vars)
        bounds_max, _ = find_range(bern_coefficients_over, self.n_vars)
        bounds = [[bounds_min, bounds_max]]
        #print("BOUNDS", bounds_min.item(), bounds_max.item(), "ORDER", self.order)

        coeffs_objs = relu_monom_coeffs(self.order, bounds)  
        relu_coefficients_over, relu_coefficients_under = coeffs_objs.get_monom_coeffs_layer()
        #print("COEF OVER", relu_coefficients_over)
        #print("COEF UNDER", relu_coefficients_under)


        #print("BEFORE BERN UNDER", bern_coefficients_under)
        relu_under = relu(bern_coefficients_under, self.n_vars, relu_coefficients_under[0])
        relu_over = relu(bern_coefficients_over, self.n_vars, relu_coefficients_over[0])
        print("Term length", len(relu_under), len(relu_over))
        #print(relu_under, relu_over)

        return relu_under, relu_over

class LayerModule(torch.nn.Module):

    def __init__(self, n_vars, weights, biases, size, order):
        super().__init__()
        weights = weights.T
        self.nodes = [NodeModule(n_vars, order, weights[i], biases[i]) for i in range(size)]

    def forward(self, inputs_under, inputs_over):
        results_under = []
        results_over = []
        for node in self.nodes:
            res_under, res_over = node(inputs_under, inputs_over)
            results_under.append(res_under)
            results_over.append(res_over)
        return results_under, results_over

class NetworkModule(torch.nn.Module):

    def __init__(self, n_vars, weights, biases, sizes, order):
        super().__init__()
        self.layers = [LayerModule(n_vars, weights[i], biases[i], sizes[i+1], order) for i in range(len(sizes)-1)]

    def forward(self, inputs):
        inputs_over = inputs
        inputs_under = inputs
        for i, layer in enumerate(self.layers):
            print(i)
            inputs_under, inputs_over = layer(inputs_under, inputs_over)
        return inputs_under, inputs_over



def convert_to_dense(l, size, n_vars, degree):
    result = []
    for i in range(size):
        x = bern_scaling_list(l[i])
        x = degree_elevation_list(x, n_vars, degree)
        x = bern_unscaling_list(x)
        result.append(combine(x, n_vars))
    return torch.stack(result)

def generate_input(intervals, n_vars, device='cuda'):
    result = []
    for i in range(n_vars):
        x = torch.ones((n_vars, 2), dtype=torch.float32, device=device)
        x[i] = torch.tensor(intervals[i])
        result.append([x])
    return result

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #np.random.seed(0)

    in_dim = 1
    hidden_dim = 2
    out_dim = 2
    n_vars = 2
    order = 2

    weights1 = torch.tensor(np.random.rand(in_dim, hidden_dim), dtype=torch.float32)
    weights2 = torch.tensor(np.random.rand(hidden_dim, out_dim), dtype=torch.float32)
    biases1 = torch.tensor(np.random.rand(hidden_dim), dtype=torch.float32)
    biases2 = torch.tensor(np.random.rand(out_dim), dtype=torch.float32)
    print("OLD")
    mod = OldLayerModule(n_vars, [1, 1], weights1.cuda(), biases1.cuda(), hidden_dim, order, 0)
    mod2 = OldLayerModule(n_vars, [1, 1], weights2.cuda(), biases2.cuda(), out_dim, order, 0)

    """
    in1 = torch.tensor([[[5.5, 4.25], [7., 4.5]], [[0.25, 0.25], [0.25, 0.25]]]).cuda()
    in2 = torch.tensor([[[3., 1.], [6., 2.]], [[2., 7.], [2., 7.]]])
    """
    in1 = torch.tensor([[[0.25, 0.25], [0.25, 0.25]]]).cuda()
    in2 = torch.tensor([[[3., 1.], [6., 2.]]])
    #result_under_orig, result_over_orig = mod2(*mod(in1.cuda(), in2.cuda()))
    result_under_orig, result_over_orig = mod(in1.cuda(), in2.cuda())
    print("UNDER", result_under_orig)
    print("OVER", result_over_orig)

    print("\nNEW")
    in_over = [[torch.tensor([[1., 2.], [3., 1.]])]]
    in_under = [[torch.tensor([[0.5, 0.5], [0.5, 0.5]])]]
    mod = LayerModule(n_vars, weights1, biases1, hidden_dim, order)
    mod2 = LayerModule(n_vars, weights2, biases2, out_dim, order)
    #result_under, result_over = mod2(*mod(in_under, in_over))
    result_under, result_over = mod(in_under, in_over)
    #print("UNDER", result_under)
    #print("OVER", result_over)
    print("UNDER", convert_to_dense(result_under, out_dim, 2, 2))
    print("OVER", convert_to_dense(result_over, out_dim, 2, 2))
    dense_under = convert_to_dense(result_under, out_dim, n_vars, result_under_orig[0].shape[0]-1)
    dense_over = convert_to_dense(result_over, out_dim, n_vars, result_under_orig[0].shape[0]-1)
    if not torch.allclose(dense_under, result_under_orig):
        print("UNDER DOES NOT MATCH")
    elif not torch.allclose(dense_over, result_over_orig):
        print("OVER DOES NOT MATCH")
    else:
        print("MATCHED")

    #print(generate_input([[0, 1], [0, 1]], n_vars))
