import torch
from poly_utils_cuda import *
from utils import *
from network_modules_batches import *

import sys
sys.path.append('/home/wael/bernstein_gpu_codes/BERN-NN-Implicit_rep/src/BERN-NN-Valen')
from torch_modules import degree_elevation as degree_elevation_old
from torch_modules import bern_coeffs_inputs, SumModule, PowerModule


###### cartesian product function ###########
def compute_cartesian_product(tensor):
    # Ensure the tensor is 2D
    assert tensor.dim() == 2, "Input needs to be a 2D tensor"

    # Get the number of rows and columns
    n, m = tensor.size()

    # Start with a 1D tensor of ones; this will serve as the accumulator
    # We're using ones because we'll be doing multiplication (neutral element)
    result = torch.ones([1], dtype=tensor.dtype)  # starting with a single 1

    # Now, we'll iteratively compute the Cartesian product
    for i in range(n):
        # Select the row
        row = tensor[i]

        # Compute the outer product with the accumulator
        result = torch.outer(result, row).flatten()  # 'outer' stands for 'outer product'
    # After computing the Cartesian product, we need to reshape the result back to n dimensions
    # The total number of elements would be m^n, not m*n.
    result = result.reshape([m] * n)

    return result





###### from implicit to dense function ###########
### given a 2D tensor T and an integer t, chunck T into t tensors of size T.size(0)/t and do the cartesian among the rows of each tensor and sum the results to get a dense tensor
### T: 2D tensor
### t: number of tensors to be created
def implicit_to_dense(T, t):
    ### get the number of rows in T
    n = T.size(0)
    ### get the number of columns in T
    m = T.size(1)
    ### get the number of rows in each tensor
    n1 = n // t
    ### create a list of tensors
    tensors_list = []
    ### loop over the tensors
    for i in range(t):
        ### get the current tensor
        tensor = T[i * n1 : (i + 1) * n1, :]
        ### get the cartesian product of the rows of the current tensor
        tensor = compute_cartesian_product(tensor)
        ### append the current tensor to the list
        tensors_list.append(tensor)
    ### sum the tensors in the list
    res = sum(tensors_list)
    return res




# ### test cartesian_product function
# tensor = torch.tensor([[2., 2.], [1., 1.], [2., 2.], [2., 2]])
# res = compute_cartesian_product(tensor)
# print(res)


# ### test implicit_to_dense function
# tensor = torch.tensor([[2., 2.], [1., 1.], [2., 2.], [2., 2]])
# res = implicit_to_dense(tensor, 2)
# print(res)




# # ### test degree_elevation function from BERN-NN-IBF and compare it with the one from BERN-NN-Valen
# n = 3
# tA = 4
# degree = torch.tensor([2, 2, 2])
# new_degree = torch.tensor([3, 4, 2])
# pA = torch.randn(tA * n, torch.max(degree) + 1)

# ### from BERN-NN-IBF
# res1 = degree_elevation(pA, n, tA, degree, new_degree)
# res1 = remove_zero_rows(res1)
# res1 = implicit_to_dense(res1, tA)
# print(res1)
# ### from BERN-NN-Valen
# pA_dense = implicit_to_dense(pA, tA)
# res2 = degree_elevation_old(pA_dense, new_degree)
# print(res2)
# ### compute the absolute difference between the two results and check if it is zero 
# res = torch.sum(torch.abs(res1 - res2))
# print(res)

# ### test mult_with_constant function from BERN-NN-IBF and compare it with the one from BERN-NN-Valen
# n = 3
# tA = 40
# degree = torch.tensor([2, 2, 2])
# pA = torch.randn(tA * n, torch.max(degree) + 1)
# const = torch.tensor([5.])
# ### from BERN-NN-IBF
# res1 = mult_with_constant(n, pA, const)
# res1 = remove_zero_rows(res1)
# res1 = implicit_to_dense(res1, tA)
# # print(res1)
# ### from BERN-NN-Valen
# pA_dense = implicit_to_dense(pA, tA)
# res2 = const * pA_dense
# # print(res2)
# ### compute the absolute difference between the two results and check if it is zero
# res = torch.max(torch.abs(res1 - res2))
# print(res)

# # ### test add_with_constant function from BERN-NN-IBF and compare it with the one from BERN-NN-Valen
# n = 2
# # tA = 30
# # degree = torch.tensor([2, 5, 7])
# # pA = torch.randn(tA * n, torch.max(degree) + 1)
# pA = torch.tensor([[-0.4894,  0.4894],
#         [ 1.0000,  1.0000],
#         [ 1.1204,  1.1204],
#         [-1.0000,  1.0000],
#         [ 0.0000,  0.0000],
#         [ 1.0000,  1.0000]])
# const = torch.tensor([1.6098])
# ### from BERN-NN-IBF
# # print(pA)
# res1 = add_with_constant(n, pA, const)
# print(res1)
# # res1 = remove_zero_rows(res1)
# # res1 = implicit_to_dense(res1, tA + 1)
# # # print(res1)
# # ### from BERN-NN-Valen
# # pA_dense = implicit_to_dense(pA, tA)
# # res2 = const + pA_dense
# # # print(res2)
# # ### compute the absolute difference between the two results and check if it is zero
# # res = torch.max(torch.abs(res1 - res2))
# # print(res)



# ### test generate-inpupt function from BERN-NN-IBF and compare it with the one from BERN-NN-Valen
# n = 10
# i = 7
# intervals = torch.tensor([2, 3] * n)
# ### from BERN-NN-IBF
# res1 = generate_inputs(n, intervals, device='cpu')
# res1 = implicit_to_dense(res1[i], 1)
# # print(res1)
# ### from BERN-NN-Valen
# res2 = bern_coeffs_inputs(n, intervals)
# # print(res2)
# ### compute the absolute difference between the two results and check if it is zero
# res = torch.max(torch.abs(res1 - res2[i]))
# print(res)


# ### test sum_2_polys function from BERN-NN-IBF and compare it with the one from BERN-NN-Valen
# n = 3
# tA = 30
# tB = 20
# degree_A = torch.tensor([4, 4, 4])
# degree_B = torch.tensor([20, 20, 20])
# pA = torch.randn(tA * n, torch.max(degree_A) + 1)
# pB = torch.randn(tB * n, torch.max(degree_B) + 1)
# ### from BERN-NN-IBF
# res1 = sum_2_polys(n, pA, tA, degree_A, pB, tB, degree_B)
# # res1 = remove_zero_rows(res1)
# res1 = implicit_to_dense(res1, tA + tB)
# # print(res1)
# ### from BERN-NN-Valen
# pA_dense = implicit_to_dense(pA, tA)
# pB_dense = implicit_to_dense(pB, tB)
# sum_module = SumModule(n, degree_A, degree_B, 1)
# res2 = sum_module(pA_dense, pB_dense)
# # print(res2)
# ### compute the absolute difference between the two results and check if it is zero
# res = torch.max(torch.abs(res1 - res2))
# print(res)


# # # ### test poly_pow_2 function from BERN-NN-IBF and compare it with the one from BERN-NN-Valen
# n = 3
# tA = 6
# degree_A = torch.tensor([4] * n)
# pA = torch.randn(tA * n, torch.max(degree_A) + 1)
# ### from BERN-NN-IBF
# res1 = poly_pow_2(n, pA, tA, degree_A)
# tA_new = res1.size(0) // n
# # res1 = mult_2_polys(n, pA, tA, degree_A,  pA, tA, degree_A)
# # res1 = remove_zero_rows(res1)
# res1 = implicit_to_dense(res1, res1.size(0) // n)
# # print(res1)
# ### from BERN-NN-Valen
# pA_dense = implicit_to_dense(pA, tA)
# power_module = PowerModule(n, degree_A, 2, 1)
# res2 = power_module(pA_dense)
# # print(res2)
# ### compute the absolute difference between the two results and check if it is zero
# res_diff = torch.abs(res1 - res2)
# # ### chunk res_diff to tA_new sub-tensors
# # res_diff = torch.chunk(res_diff, tA_new, dim=0)
# # print(res_diff)
# ### find the maximu for each sub-tensor
# # res = torch.stack([torch.max(res_diff[i]) for i in range(tA_new)])
# res = torch.max(torch.abs(res1 - res2))
# # print(res)





