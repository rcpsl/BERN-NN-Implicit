import torch

# def split_and_select(pA, tA, I):
#     # Calculate the number of rows in each subtensor
#     rows_per_subtensor = pA.size(0) // tA

#     # Reshape the tensor into tA subtensors
#     subtensors = pA.split(rows_per_subtensor)

#     # Stack the subtensors along a new dimension to select rows
#     stacked_subtensors = torch.stack(subtensors)
    
#     print(stacked_subtensors)
#     # Use tensor indexing to select rows based on indices
#     selected_rows = stacked_subtensors[:, I]
#     print(selected_rows)
#     # Reshape the result to a 2D tensor
#     pAmod = selected_rows.view(-1, selected_rows.size(-1))

#     return pAmod

# # Example usage
# pA = torch.tensor([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9],
#                    [10, 11, 12],
#                    [0, 1, 2],
#                    [1, 1, 1]])

# tA = 2
# I = torch.tensor([1, 2])

# pAmod = split_and_select(pA, tA, I)
# print(pAmod)


# def one_dim_coeffs(degree):
#     """
#     generates multinomial coefficients as a tensor in 1D
#     degree: degree for multinomial coefficients
#     """
#     N = torch.tensor(degree)
#     R = torch.arange(degree + 1)
#     a = torch.lgamma(N + 1)
#     b = torch.lgamma(N-R+1)
#     c = torch.lgamma(R+1)
#     return torch.exp(a-b-c)


# def generate_binomial_coefficients(I):
#     max_i = torch.max(I)
#     idx = torch.arange(max_i+1)
#     print(idx)
#     combinations = torch.combinations(idx, I[:, None])
#     res = torch.prod(combinations, dim=2)
#     return res

# def generate_ranges(L):
#     ranges = [torch.arange(i + 1) for i in L]
#     return torch.nn.utils.rnn.pad_sequence(ranges, batch_first=True, padding_value=0) 

# def generate_binom_coeffs(L):
#     ranges = [torch.arange(i + 1) for i in L]
#     N = L.reshape([len(L), 1])
#     R = torch.nn.utils.rnn.pad_sequence(ranges, batch_first=True, padding_value = -1) 
#     A = torch.lgamma(N + 1)
#     print(N)
#     print(R)
#     B = torch.lgamma(N - R + 1)
#     C = torch.lgamma(R + 1)
#     return torch.exp(A - B - C)



# # Example usage
# L = torch.tensor([2, 4, 3])
# result = generate_binom_coeffs(L)
# print(result)



# t1 = torch.tensor([1, 2])
# t2 = torch.tensor([1, 2, 1])

# t1 = t1.reshape(1, -1).reshape(1, 1, -1)
# t2 = t2.reshape(1, -1)
# t2 = torch.flip(t2, dims=(1,)).reshape(1, 1, -1)

# print(torch.nn.functional.conv1d(t1, t2, padding = 2).flatten())

# import numpy as np
# print(np.convolve([2, 3], [1, 1]))



# a = torch.tensor([[4, 5, 6], [8, 6, 9], [10, 2, 3], [9, 6, 1]])

# a = torch.nn.functional.pad(a, (0, 0))
# print(a)



# T1 = torch.tensor([[1, 2, 3],
#                    [4, 5, 6],
#                    [7, 8, 9]])

                   


# T2 = torch.tensor([[10, 11, 12],
#                    [13, 14, 15]])

# I = torch.tensor([1, 2])

# T1[I , :] = T2

# indices = torch.where(torch.all(T1.unsqueeze(1) == T2, dim=2))[0]
# print(indices)







# T1 = torch.tensor([1.0, 2.0, 0.0, 4.0])
# T2 = torch.tensor([0.5, 0.0, 3.0, 2.0])

# result =  torch.where(T2 != 0, torch.div(T1, T2), 0)

# print(result)




# pA = torch.tensor([[-1.2830, -2.5659],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [-1.4432, -0.0000],
#         [ 3.0000,  4.0000],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [-0.0000, -0.0000],
#         [ 1.0000,  0.0000],
#         [ 7.0000,  9.0000],
#         [ 1.0000,  0.0000],
#         [-0.3170, -0.0000],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 5.0000,  6.0000],
#         [ 0.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 0.0000,  0.0000],
#         [ 3.0000,  4.0000],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 0.1740,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 7.0000,  9.0000],
#         [ 1.0000,  0.0000],
#         [ 0.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 5.0000,  6.0000],
#         [-1.0000, -1.0000],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000],
#         [ 1.0000,  0.0000]])


# print(pA.size(0))
# n = 4
# tA = 9
# res = [pA[i * n : (i + 1) * n, :] for i in range(tA)]

# print(len(res))
# for i in range(len(res)):
#     print(res[i])




### perform row-wise convolution between T_C_A and T_C_B
n = 10
# T_C_A = torch.tensor([[1.0, 2.0, 8.0, 4.0, 5.],[3.0, 4.0, 7.0, 3.0, 5.]])
# T_C_B = torch.tensor([[0.5, 0.0, 3.0, 2.0, 5.],[1.0, 0.0, 5.0, 0.0, 5.]])
T_C_A = torch.randn(n, 5)
T_C_B = torch.randn(n, 5)
res1 = [torch.nn.functional.conv1d(T_C_A[i, :].reshape(1, -1).reshape(1, 1, -1),      torch.flip(T_C_B[i, :].reshape(1, -1), dims=(1,)).reshape(1, 1, -1), padding = T_C_B.size(1) - 1   ).flatten()             for i in range(n)]
res1 = torch.stack(res1) 
# print(res1)


# # Assuming T_C_A and T_C_B are both 2D tensors of shape (n, m)
# n, m = T_C_A.shape
# # Reshape T_C_A and T_C_B to have an additional channel dimension
# T_C_A = T_C_A.unsqueeze(1)  # Shape: (n, 1, m)
# T_C_B = T_C_B.unsqueeze(1)  # Shape: (n, 1, m)
# # Flip T_C_B
# T_C_B_flipped = torch.flip(T_C_B, dims=(2,))  # Shape: (n, 1, m)
# # Perform convolution
# res = torch.nn.functional.conv1d(T_C_A, T_C_B_flipped, padding  = m - 1).flatten(1)  # Shape: (n, m)
# res = res.reshape(-1, 2 * m - 1)
# res = res[torch.arange(0, res.size(0), step = n + 1)]
# print(res)


# # Assuming T_C_A and T_C_B are both 2D tensors of shape (n, m)
n, m = T_C_A.shape
# Reshape T_C_A and T_C_B to have an additional channel dimension
T_C_A = T_C_A.view((1, n, m))  # Shape: (1, n, m)
T_C_B = T_C_B.view((n, 1, m))  # Shape: (n, 1, m)
# Flip T_C_B
T_C_B_flipped = torch.flip(T_C_B, dims=(2,))  # Shape: (1, n, m)
# Perform convolution
res2 = torch.nn.functional.conv1d(T_C_A, T_C_B_flipped, padding  = m - 1, groups= n).flatten(1)  # Shape: (n, m)
res2 = res2.reshape(-1, 2 * m - 1)
print(torch.max(torch.abs(res1 - res2)))



# # Assuming T_C_A and T_C_B are both 2D tensors of shape (n, m)
# n, m = T_C_A.shape
# # Reshape T_C_A and T_C_B to have an additional channel dimension
# T_C_A = T_C_A.reshape((1, n, m))  # Shape: (1, n, m)
# T_C_B = T_C_B.reshape((1, n, m))  # Shape: (1, n, m)
# # Flip T_C_B
# T_C_B_flipped = torch.flip(T_C_B, dims=(2,))  # Shape: (1, n, m)
# # Perform convolution
# res = torch.nn.functional.conv1d(T_C_A, T_C_B_flipped, padding  = m - 1).flatten(0)  # Shape: (n, m)
# res = res.reshape(-1, 2 * m - 1)
# res = res[torch.arange(0, res.size(0), step = n + 1)]
# print(res)


# def generate_binom_coeffs(L):
#     ranges = [torch.arange(i + 1) for i in L]
#     N = L.reshape([len(L), 1])
#     R = torch.nn.utils.rnn.pad_sequence(ranges, batch_first=True, padding_value = -1) 
#     A = torch.lgamma(N + 1)
#     B = torch.lgamma(N - R + 1)
#     C = torch.lgamma(R + 1)
#     return torch.exp(A - B - C)


# def mult_2_terms(n, TA, C_A, TB, C_B, C_AmulB):

#     ### perform multpilcation point-wise between TA and CA; TB and CB
#     T_C_A = torch.mul(TA, C_A)
#     T_C_B = torch.mul(TB, C_B)

#     # ### 1) perform row-wise convolution between T_C_A and T_C_B   using for loop
#     # res = [torch.nn.functional.conv1d(T_C_A[i, :].reshape(1, -1).reshape(1, 1, -1),      torch.flip(T_C_B[i, :].reshape(1, -1), dims=(1,)).reshape(1, 1, -1), padding = T_C_B.size(1) - 1   ).flatten()             for i in range(n)]
#     # ### stack the rows
#     # res = torch.stack(res) 

#     ### 2) perform row-wise convolution between T_C_A and T_C_B   without using for loop
#     # Assuming T_C_A and T_C_B are both 2D tensors of shape (n, m)
#     n, m = T_C_A.shape
#     # Reshape T_C_A and T_C_B to have an additional channel dimension
#     T_C_A = T_C_A.unsqueeze(1)  # Shape: (n, 1, m)
#     T_C_B = T_C_B.unsqueeze(1)  # Shape: (n, 1, m)
#     # Flip T_C_B
#     T_C_B_flipped = torch.flip(T_C_B, dims=(2,))  # Shape: (n, 1, m)
#     # Perform convolution
#     res = torch.nn.functional.conv1d(T_C_A, T_C_B_flipped, padding=m - 1).flatten(1)  # Shape: (n, m)
#     res = res.reshape(-1, 2 * m - 1)
#     res = res[torch.arange(0, res.size(0), step = n + 1)]
    
#     ### perform row-wise division between res and C_AmulB for non-zero elements of C_AmulB
#     res =  torch.where(C_AmulB != 0, torch.div(res, C_AmulB), torch.tensor(0.))

#     return res


# def mult_2_terms_batched(n, TA, C_A, TB, C_B, C_AmulB):
#     # Perform element-wise multiplication with broadcast
#     T_C_A = TA * C_A.view(1, C_A.size(0),C_A.size(1))  # Shape: (1, tA, m)
#     T_C_B = TB * C_B.view(1, C_B.size(0),C_B.size(1))  # Shape: (1, tB, m)
#     C_AmulB = C_AmulB.view(1, C_AmulB.size(0),C_AmulB.size(1))  # Shape: (1, tA, 2m-1

#     # Perform batched convolution
#     res = torch.nn.functional.conv1d(T_C_A, torch.flip(T_C_B, dims=(2,)),
#                                       padding=TB.shape[1] - 1)  # Shape: (1, tA, 2m-1)

#     # Take only the required elements from res
#     res = res[:, :, torch.arange(0, res.shape[2], step=n + 1)]
#     print(res)
#     # Perform row-wise division
#     res = torch.where(C_AmulB != 0, torch.div(res, C_AmulB), torch.tensor(0.))

#     return res

# ### test mul_2_terms function
# def mult_2_polys(n, pA, tA, degree_A,  pB, tB, degree_B):

#     ### compute 2D binomial coefficients for degree_A and degree_B
#     C_A = generate_binom_coeffs(degree_A)
#     C_B = generate_binom_coeffs(degree_B)

#     ### compute 2D binomial coefficients for degree_mul = degree_A + degree_B
#     degree_mul = degree_A + degree_B
#     C_AmulB = generate_binom_coeffs(degree_mul)

#     ### perform multiplication between each term of pA to all the terms of pB
#     res = [mult_2_terms(n, pA[i * n : (i + 1) * n, :], C_A, pB[j * n : (j + 1) * n, :], C_B, C_AmulB) for j in range(tB)   for i in range(tA)]
#     # res = mult_2_terms_batched(n, pA, C_A, pB, C_B, C_AmulB)
#     print(res)
#     ### stack the rows and reshape the final result to (number_of rows, torch.max(degree_mul) + 1)
#     res = torch.stack(res).reshape(-1, torch.max(degree_mul) + 1)

#     return res

# n = 2
# pA = torch.tensor([[1.0, 2.0, 8.0],[3.0, 4.0, 7.0]])
# tA = 1
# degree_A = torch.tensor([2, 2])
# res = mult_2_polys(n, pA, tA, degree_A,  pA, tA, degree_A)
# print(res)







# # Example usage
# I = torch.tensor([1, 3, 4])  # Indices to remove
# T = torch.tensor([[1, 2, 3, 4, 7],
#                   [5, 6, 7, 8, 7],
#                   [9, 10, 11, 12, 7]])

# # Remove columns specified by indices in I
# T_filtered = T[:, [i for i in range(T.shape[1]) if i not in I]]

# print(T_filtered)




# res = torch.tensor([[1., 0.]])
# print(res[0, :])
# C_diff = torch.tensor([[1., 1.]])
# l_diff = torch.tensor([0., 1.])
# padding = torch.max(l_diff).int().item()
# print('padding', padding)
# result = [torch.nn.functional.conv1d(res[i, :].reshape(1, -1).reshape(1, 1, -1),      torch.flip(C_diff[i, :].reshape(1, -1), dims=(1,)).reshape(1, 1, -1), padding =  torch.max(l_diff).int().item()  ).flatten()             for i in range(res.size(0) )]
# print(result)

# import numpy as np
# print(np.convolve([1, 0], [1, 1]))


# import torch
# import concurrent.futures
# import time

# # Define your function that works with PyTorch tensors
# def your_function(arg):
#     device = torch.device('cuda')
#     tensor = torch.tensor(arg, device=device)
#     result = tensor * 2
#     return result

# # List of arguments for your function
# args = [[1, 2, 3, 4, 5] for _ in range(700000)]


########################################################################################
# start_time = time.time()
# # Create a ThreadPoolExecutor
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     # Map the function to the arguments in parallel
#     results = list(executor.map(your_function, args))

# print(f"--- {time.time() - start_time} seconds ---")
########################################################################################

# start_time = time.time()
# res = [your_function(args[i]) for i in range(len(args))]
# print(f"--- {time.time() - start_time} seconds ---")

# print(results)

########################################################################################

import torch.nn as nn

# # Wrap your function with DataParallel
# your_function_parallel = nn.DataParallel(your_function)

# # # Move the input data to the GPU
# # args = [arg.to('cuda') for arg in args]

# # Run the function in parallel on all available GPUs
# results = your_function_parallel(*args)

# print(results)

########################################################################################

import torch.nn as nn
import time

# Define your function that works with PyTorch tensors
# def your_function(args):
#     result = args[0] + args[1]
#     return result


# class mult_2_terms(torch.nn.Module):
#     def __init__(self, n):
#         super().__init__()
#         self.n = 1

#     def forward(self, args):
#         print('#####################################################')
#         print('args', args)
#         output = your_function(args) 
#         return output

 

# # List of arguments for your model
# args = torch.tensor([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5] for _ in range(20)])

# print(args)


# time1 = time.time()
# # # Move the input data to the GPU
# device = torch.device('cuda')
# args = args.to(device)
# # define the model
# model = mult_2_terms(5) 
# # Wrap your function with DataParallel
# model_parallel = nn.DataParallel(model) 
# # Move the model to the GPU
# model_parallel.to(device)

# # Run the function in parallel on all available GPUs
# results = model_parallel(args)
# print(f"--- {time.time() - time1} seconds ---")

# print(results)



# pA = torch.tensor([[-1.5099e-02,  1.5099e-02, -1.5099e-02],
#         [ 1.0000e+00,  1.0000e+00,  1.0000e+00],
#         [ 1.0000e+00,  1.0000e+00,  1.0000e+00],
#         [-1.2570e-02, -1.2434e-10,  1.2570e-02],
#         [-1.0000e+00,  0.0000e+00,  1.0000e+00],
#         [ 1.0000e+00,  1.0000e+00,  1.0000e+00]])
# pB = torch.tensor([[-1.9140e-01,  1.9140e-01, -1.9140e-01],
#         [ 1.0000e+00,  1.0000e+00,  1.0000e+00],
#         [ 1.0000e+00,  1.0000e+00,  1.0000e+00],
#         [ 1.0091e-01, -1.3028e-09, -1.0091e-01],
#         [-1.0000e+00,  0.0000e+00,  1.0000e+00],
#         [ 1.0000e+00,  1.0000e+00,  1.0000e+00]])

# tA = 2
# tB = 2
# n = 3
# res = [torch.cat(((pA[i * n , :] + pB[j * n, :]).unsqueeze(0), pA[i * n + 1 : (i + 1) * n, :])) for j in range(tB)   for i in range(tA) if torch.equal(pA[i * n + 1 : (i + 1) * n, :], pB[j * n + 1 : (j + 1) * n, :])]
# res = torch.cat(res, dim=0)
# print(res)

# def process_TA(TA, tA):
#         ###create TA_mod1 = repeated every n_vars rows of TA tA times along the first dimension
#         TA_chunks = torch.chunk(TA, tA)
#         # print('len(TA_chunks)', len(TA_chunks))
#         TA_mod1 = [TA_chunks[i].repeat(tA - i, 1) for i in  range(tA)]
#         TA_mod1 = torch.cat(TA_mod1, dim=0)
#         return TA_mod1


# def process_tensor_optimized(TA, tA, n):
#     # Create a tensor that represents the repeat pattern.
#     repeat_counts = torch.arange(start=tA, end=0, step=-1, device=TA.device)  # [tA, tA-1, ..., 1]

#     # Create a tensor of indices that represent the positions.
#     indices = torch.arange(start=0, end=tA, device=TA.device)  # [0, 1, ..., tA-1]

#     # Repeat each index (tA - index) times. This creates a pattern of indices.
#     index_tensor = torch.repeat_interleave(indices, repeat_counts)

#     # Adjust the indices to account for 'n' rows in each chunk.
#     # Each index needs to be converted to 'n' row indices, resulting in a flat array of row indices.
#     expanded_index_tensor = index_tensor * n
#     row_indices = expanded_index_tensor.view(-1, 1).repeat(1, n) + torch.arange(n, device=TA.device)

#     # Flatten the row_indices for gathering operation.
#     flat_row_indices = row_indices.view(-1)

#     # Gather the rows from the original tensor.
#     result_tensor = torch.index_select(TA, 0, flat_row_indices)

#     return result_tensor









# # Test the function with your parameters
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# n = 5
# pA = torch.tensor([[1, 10, 5] for _ in range(n)])
# tA = 30000
# pA = pA.repeat(tA, 1)

# # time_start = time.time()
# # TA_mod1 = process_TA(pA, tA)
# # time_end = time.time()
# # print('time TA_mod1', time_end - time_start)

# time_start = time.time()
# TA_mod2 = process_tensor_optimized(pA, tA, n)
# time_end = time.time()
# print('time TA_mod2', time_end - time_start)

# # The difference should be a tensor with all zeros, indicating identical results from both methods.
# difference = TA_mod2 - TA_mod1
# print('Difference:', torch.abs(difference).sum())  # This should be 0 if both tensors are identical.





# def modify_tensor(TA, tA, n_vars):
#     TA_mod2 = [TA[i * n_vars:].clone() for i in  range(tA)]
#     for i in range(len(TA_mod2)):
#         chunk = TA_mod2[i]
#         chunk[torch.arange(n_vars, chunk.size(0), n_vars)] *= 2

#     TA_mod2 = torch.cat(TA_mod2, dim=0)
#     return TA_mod2 



# # Test the function with your parameters
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# n = 3
# pA = torch.tensor([[1, 1, 1] for _ in range(n)])
# tA = 4
# pA = pA.repeat(tA, 1)
# # print(pA)
# res = modify_tensor(pA, tA, n)
# print(res)