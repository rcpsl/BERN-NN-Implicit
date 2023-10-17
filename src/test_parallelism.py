
import torch
import torch.nn as nn
import time

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:516"

###################################################
### given a tensor of degrees L = tensor([L1,...,Ln]),
###   it outputs 2D 
### tensor([[(L1 choose 0),...,(L1 choose L1)],
###             .
###             .
###         [(Ln choose 0),...,(Ln choose Ln)]])
###################################################
def generate_binom_coeffs(L):
    ranges = [torch.arange(i + 1) for i in L]
    N = L.reshape([len(L), 1])
    R = torch.nn.utils.rnn.pad_sequence(ranges, batch_first=True, padding_value = -1) 
    A = torch.lgamma(N + 1)
    B = torch.lgamma(N - R + 1)
    C = torch.lgamma(R + 1)
    return torch.exp(A - B - C)


###################################################
### performs multiplicaion between 2 2D terms  
###  tensors TA and TB
###################################################
def mult_2_terms(n, TA, C_A, TB, C_B, C_AmulB):

    ### perform multpilcation point-wise between TA and CA; TB and CB
    T_C_A = torch.mul(TA, C_A)
    T_C_B = torch.mul(TB, C_B)

    # ### 1) perform row-wise convolution between T_C_A and T_C_B   using for loop
    # res = [torch.nn.functional.conv1d(T_C_A[i, :].reshape(1, -1).reshape(1, 1, -1),      torch.flip(T_C_B[i, :].reshape(1, -1), dims=(1,)).reshape(1, 1, -1), padding = T_C_B.size(1) - 1   ).flatten()             for i in range(n)]
    # ### stack the rows
    # res = torch.stack(res) 

    ### 2) perform row-wise convolution between T_C_A and T_C_B   without using for loop
    # Assuming T_C_A and T_C_B are both 2D tensors of shape (n, m)
    n, m = T_C_A.shape
    # print('nnnnnnnnnnnnnnnnn', n)
    # Reshape T_C_A and T_C_B to have an additional channel dimension
    T_C_A = T_C_A.unsqueeze(1)  # Shape: (n, 1, m)
    T_C_B = T_C_B.unsqueeze(1)  # Shape: (n, 1, m)
    # Flip T_C_B
    T_C_B_flipped = torch.flip(T_C_B, dims=(2,))  # Shape: (n, 1, m)
    # Perform convolution
    res = torch.nn.functional.conv1d(T_C_A, T_C_B_flipped, padding = m - 1).flatten(1)  # Shape: (n, m)
    res = res.reshape(-1, 2 * m - 1)
    # print(res.shape)
    res = res[torch.arange(0, res.size(0), step = n + 1)]
    
    ### perform row-wise division between res and C_AmulB for non-zero elements of C_AmulB
    # print(res)
    # print(res.shape)
    # print(C_AmulB.shape)
    res =  torch.where(C_AmulB != 0, torch.div(res, C_AmulB), torch.tensor(0.))

    return res





###################################################
### performs multiplicaion between 2 2D tensor
###  polynomials pA and pB.
### n: number of variables
### tA: number of terms in pA
### degree_A: degree of pA
### tB: number of terms in pB
### degree_B: degree of pB
###################################################

def mult_2_polys(n, pA, tA, degree_A,  pB, tB, degree_B):

    ### compute 2D binomial coefficients for degree_A and degree_B
    C_A = generate_binom_coeffs(degree_A)
    C_B = generate_binom_coeffs(degree_B)

    ### compute 2D binomial coefficients for degree_mul = degree_A + degree_B
    degree_mul = degree_A + degree_B
    C_AmulB = generate_binom_coeffs(degree_mul)

    ### perform multiplication between each term of pA to all the terms of pB
    res = [mult_2_terms(n, pA[i * n : (i + 1) * n, :], C_A, pB[j * n : (j + 1) * n, :], C_B, C_AmulB) for j in range(tB)   for i in range(tA)]
    
    # print(res)
    ### stack the rows and reshape the final result to (number_of rows, torch.max(degree_mul) + 1)
    res = torch.stack(res).reshape(-1, torch.max(degree_mul).int().item() + 1)
    # print('resresresres', res.shape)

    return res




# #### define Mult_2_terms_over_gpus class that parallelizes the computations of mult_2_terms over multiple GPUs
# class Mult_2_terms_over_gpus(torch.nn.Module):
#     def __init__(self, n, tA, CA, CB, tB, C_AmulB):
#         super().__init__()
#         self.n = n
#         self.tA = tA
#         self.CA = CA
#         self.CB = CB
#         self.tB = tB
#         self.C_AmulB = C_AmulB

#     def forward(self, inputs):
#         res = mult_2_terms(self.n, , self.C_A, TB, self.C_B, self.C_AmulB)



###################################################
### performs power 2 of a 2D polynomial tensor TA  \
### tA: number of terms in pA
### degree_A: degree of pA
###################################################
def poly_pow_2(n_vars, TA, tA, degree_A):
    
    ###create TA_mod1 = repeated every n_vars rows of TA tA times along the first dimension
    TA_mod1 = torch.chunk(TA, n_vars)
    TA_mod1 = [chunk.repeat(tA, 1) for chunk in TA_mod1]
    TA_mod1 = torch.cat(TA_mod1, dim=0)
    ### create TA_mod2 = reapeted TA tA times along the first dimension
    TA_mod2 = TA.repeat(tA, 1)
    ### compute 2D binomial coefficients for degree_A 
    C_A = generate_binom_coeffs(degree_A)
    ### repeat C_A tA ** 2 times along the first dimension
    C_A = C_A.repeat(tA ** 2, 1)

    ### compute 2D binomial coefficients for degree_mul = 2 * degree_A 
    degree_mul = 2 * degree_A 
    C_AmulA = generate_binom_coeffs(degree_mul)
    ### repeat C_AmulA tA^2 times along the first dimension
    C_AmulA = C_AmulA.repeat(tA ** 2, 1)

    ### perform multpilcation point-wise between TA_mod1 and CA; TA_mod2 and CA
    # print(TA_mod1.shape)
    # print(C_A.shape)
    T_C_A = torch.mul(TA_mod1, C_A)
    T_C_B = torch.mul(TA_mod2, C_A)
    



    # ### 2) perform row-wise convolution between T_C_A and T_C_B   without using for loop
    # # Assuming T_C_A and T_C_B are both 2D tensors of shape (n, m)
    # n_rows, m = T_C_A.shape
    # # Reshape T_C_A and T_C_B to have an additional channel dimension
    # T_C_A = T_C_A.unsqueeze(1)  # Shape: (n_rows, 1, m)
    # T_C_B = T_C_B.unsqueeze(1)  # Shape: (n_rows, 1, m)
    # # Flip T_C_B
    # T_C_B_flipped = torch.flip(T_C_B, dims=(2,))  # Shape: (n_rows, 1, m)
    # # Perform convolution
    # print(T_C_A.size(), T_C_B_flipped.size())
    # res = torch.nn.functional.conv1d(T_C_A, T_C_B_flipped, padding = m - 1)  # Shape: (n_rows, m)
    # # print(res)
    # # print('resresresres', res.shape, m, n_rows)
    # print(res.size())
    # res = res.flatten(1)

    # res = res.reshape(-1, 2 * m - 1)
    # # print(res)
    # # print('resresresres', res.shape)
    # res = res[torch.arange(0, res.size(0), step = n_rows + 1)]



    # Assuming T_C_A and T_C_B are both 2D tensors of shape (n, m)
    n_rows, m = T_C_A.shape
    # Reshape T_C_A and T_C_B to have an additional channel dimension
    T_C_A = T_C_A.view((1, n_rows, m))  # Shape: (1, n_rows, m)
    T_C_B = T_C_B.view((n_rows, 1, m))  # Shape: (n_rows, 1, m)
    # Flip T_C_B
    T_C_B_flipped = torch.flip(T_C_B, dims=(2,))  # Shape: (1, n_rows, m)
    # Perform convolution
    res = torch.nn.functional.conv1d(T_C_A, T_C_B_flipped, padding  = m - 1, groups = n_rows).flatten(1)  # Shape: (n_rows, m)
    res = res.reshape(-1, 2 * m - 1)
    # print(res)
        
    ### perform row-wise division between res and C_AmulA for non-zero elements of C_AmulA
    # print(res.shape)
    # print(C_AmulA.shape)
    res =  torch.where(C_AmulA != 0, torch.div(res, C_AmulA), torch.tensor(0.))

    return res



### test mult_2_polys and poly_pow_2 functions and compare their results and running times
torch.set_default_tensor_type('torch.cuda.FloatTensor')
n = 4
pA = torch.tensor([[1, 2, 3], [5, 6, 3], [1, 2, 3], [5, 6, 3]])
# device = torch.device('cuda')
# pA = pA.to(device)
tA = 3000
pA = pA.repeat(tA, 1)
degree_A = torch.tensor([2, 2, 2, 2])
time_start = time.time()
# res1 = mult_2_polys(n, pA, tA, degree_A,  pA, tA, degree_A)
# print("time for mult_2_polys: ", time.time() - time_start)
# print(res1)
time_start = time.time()
res2 = poly_pow_2(n, pA, tA, degree_A)
print("time for poly_pow_2: ", time.time() - time_start)
# print(res1 - res2)
