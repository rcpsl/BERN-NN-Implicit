import torch
from poly_utils_cuda import mult_with_constant, add_with_constant



import sys
import ibf_dense_cpp

import find_indices_cpp


def find_indices(degree):

    sizes = (degree + 1).tolist()
    # print('sizes', sizes)
    mat = torch.ones(sizes, dtype=torch.float32)   
    indices = torch.nonzero(mat)

    return indices


def ibf_to_dense(n_vars, bern_ibf):
    # print('bern_ibf.shape', bern_ibf.shape)
    ### reshape bern_ibf to be of shape ()
    bern_ibf = torch.reshape(bern_ibf, (bern_ibf.size(0) // n_vars, n_vars, bern_ibf.size(1)))
    bern_ebf = ibf_dense_cpp.ibf_dense(bern_ibf)
    ### convert ebf to 1 column
    bern_ebf = bern_ebf.reshape(-1, 1)
    return bern_ebf

def generate_linear_ibf(n_vars, intervals, coeffs, device = 'cuda'):
    # print('coeffs', coeffs)
    inputs = []
    for i in range(n_vars + 1):

        if i != n_vars:
            ith_input = torch.ones((n_vars, 2), dtype = torch.float32, device = device)
            ith_input[i, :] = intervals[i]
            ith_input[0, :] *=  coeffs[i] 

            inputs.append(ith_input)

        else:
            ith_input = torch.ones((n_vars, 2), dtype = torch.float32, device = device)
            # print('coeffs[i]', coeffs[i])
            ith_input[0, :] *= coeffs[i] 
            inputs.append(ith_input)
    
    ### stack all inputs
    inputs = torch.cat(inputs, dim = 0)
    # print('inputs', inputs)
    return inputs



def control_points_matrix(degree, intervals):
    
    n_vars = degree.shape[0]
    # print('n_vars', n_vars)
    degree = degree.to(torch.int32)

    if n_vars < 17:
        batch_size = 3 ** 15
        indices = find_indices_cpp.find_indices(degree, batch_size)
    else:
        indices = find_indices(degree)

    intervals_diff = intervals[:, 1] - intervals[:, 0]
    lowers = intervals[:, 0]
    res = indices * intervals_diff / degree + lowers
    ones = torch.ones(res.shape[0]).reshape(-1, 1)
    return torch.hstack([res, ones]), indices



def lower_linear_ibf(dims, intervals, bern_ibf, degree):
    # print('bern_ibf.shape', bern_ibf.shape)
    # intervals = torch.tensor(intervals)
    intervals_diff = intervals[:, 1] - intervals[:, 0]
    lowers = intervals[:, 0]

    A, indices = control_points_matrix(degree, intervals)
    ### convert bern_ibf to bern_coeffs
    bern_coeffs = ibf_to_dense(dims, bern_ibf)

    AT = A.T
    AT_A = torch.matmul(AT, A)
    # print(AT.shape)
    # print(bern_coeffs.shape)
    AT_b = torch.matmul(AT, bern_coeffs)


    gamma = torch.linalg.lstsq(AT_A, AT_b, rcond=None)[0]

    # compute maximum error delta
    indices = (intervals_diff * indices) / (degree) + lowers

    cvals_star = torch.matmul(indices, gamma[:-1]) + gamma[-1]
    errors = cvals_star - bern_coeffs

    delta_plus = errors.max() / (10 ** 5)
    delta_minus = (-1.0) * errors.min() / (10 ** 5)
    gamma[-1] = gamma[-1] - delta_plus 
 
    # print('network_input', network_inputs.shape)
    # print('gamma.shape', gamma.shape)
    # print('gammagamma', len(gamma))
    lower_linear_ibf_output = generate_linear_ibf(dims, intervals, gamma) 
    # print('lower_linear_ibf_output.shape', lower_linear_ibf_output.shape)

    # cvals = torch.matmul(indices, gamma[:-1]) + gamma[-1]
    # errors =  bern_coeffs - cvals
    # delta_minus = errors.max()


    return lower_linear_ibf_output, delta_minus + delta_plus, gamma 



def upper_linear_ibf(dims, intervals, bern_ibf, degree): 



    # minus_bern_coeffs = mult_with_constant(dims, bern_ibf, -1)

    # lower_linear_ibf_output= lower_linear_ibf(dims, intervals, minus_bern_coeffs, degree)

    # upper_linear_ibf_output =   mult_with_constant(dims, lower_linear_ibf_output, -1) 
    lower_linear_ibf_output, delta, gamma = lower_linear_ibf(dims, intervals, bern_ibf, degree)
    gamma[-1] = gamma[-1] + delta
    # upper_linear_ibf_output =   add_with_constant(dims, lower_linear_ibf_output, delta )
    upper_linear_ibf_output = generate_linear_ibf(dims, intervals, gamma)


    return upper_linear_ibf_output, delta, gamma



def lower_linear_ibf_1_lin(dims, intervals, bern_ibf, degree, delta_error):
    # print('bern_ibf.shape', bern_ibf.shape)
    # intervals = torch.tensor(intervals)
    intervals_diff = intervals[:, 1] - intervals[:, 0]
    lowers = intervals[:, 0]

    A, indices = control_points_matrix(degree, intervals)
    ### convert bern_ibf to bern_coeffs
    bern_coeffs = ibf_to_dense(dims, bern_ibf)

    AT = A.T
    AT_A = torch.matmul(AT, A)
    # print(AT.shape)
    # print(bern_coeffs.shape)
    AT_b = torch.matmul(AT, bern_coeffs)


    gamma = torch.linalg.lstsq(AT_A, AT_b, rcond=None)[0]

    # compute maximum error delta
    indices = (intervals_diff * indices) / (degree) + lowers

    cvals_star = torch.matmul(indices, gamma[:-1]) + gamma[-1]
    errors = cvals_star - bern_coeffs

    delta_plus = errors.max() * 1.02
    delta_minus = (-1.0) * errors.min() * 1.02
    gamma[-1] = gamma[-1] - delta_error
    # print('delta_plus', delta_plus)
    # print('delta_minus', delta_minus)
 
    # print('network_input', network_inputs.shape)
    # print('gamma.shape', gamma.shape)
    # print('gammagamma', len(gamma))
    # lower_linear_ibf_output = generate_linear_ibf(dims, intervals, gamma) 
    # print('lower_linear_ibf_output.shape', lower_linear_ibf_output.shape)

    # cvals = torch.matmul(indices, gamma[:-1]) + gamma[-1]
    # errors =  bern_coeffs - cvals
    # delta_minus = errors.max()


    return gamma, delta_minus + delta_plus



def upper_linear_ibf_1_lin(dims, intervals, bern_ibf, degree, delta_error): 




    gamma, delta = lower_linear_ibf_1_lin(dims, intervals, bern_ibf, degree, delta_error)
    gamma[-1] = gamma[-1] + delta



    return gamma, delta





# if __name__ == "__main__":
    
#     # # Set the default tensor type to be 32-bit float on the GPU
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     ### test lower_linear_ibf
#     n_vars = 3
#     degree = torch.tensor([4, 4, 4])
#     intervals = torch.tensor([[0, 1], [0, 1], [0, 1]])


#     tA = 20
#     bern_ibf = torch.rand((n_vars, 5), device = 'cuda')
#     bern_ibf = bern_ibf.repeat(tA, 1)

#     lower_linear_ibf_output = lower_linear_ibf(n_vars, intervals, bern_ibf, degree)

#     print('lower_linear_ibf_output', lower_linear_ibf_output)
