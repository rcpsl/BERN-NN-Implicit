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




pA = torch.tensor([[-1.2830, -2.5659],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000],
        [-1.4432, -0.0000],
        [ 3.0000,  4.0000],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000],
        [-0.0000, -0.0000],
        [ 1.0000,  0.0000],
        [ 7.0000,  9.0000],
        [ 1.0000,  0.0000],
        [-0.3170, -0.0000],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000],
        [ 5.0000,  6.0000],
        [ 0.0000,  0.0000],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000],
        [ 0.0000,  0.0000],
        [ 3.0000,  4.0000],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000],
        [ 0.1740,  0.0000],
        [ 1.0000,  0.0000],
        [ 7.0000,  9.0000],
        [ 1.0000,  0.0000],
        [ 0.0000,  0.0000],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000],
        [ 5.0000,  6.0000],
        [-1.0000, -1.0000],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000],
        [ 1.0000,  0.0000]])


print(pA.size(0))
n = 4
tA = 9
res = [pA[i * n : (i + 1) * n, :] for i in range(tA)]

print(len(res))
for i in range(len(res)):
    print(res[i])

