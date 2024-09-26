import torch
import ibf_minmax_cpp


import torch

def min_max_product(T):
    # Ensure input is a float tensor (necessary for operations like multiplication)
    T = T.float()

    # Create a mask for non-zero elements and convert it to integers
    non_zero_mask = (T != 0).int()

    # Find indices of the first non-zero element in each row
    first_non_zero_idx = torch.argmax(non_zero_mask, dim=1)

    # Find indices of the last non-zero element in each row
    # We flip the tensor and the mask to use argmax for the last non-zero
    last_non_zero_idx = T.size(1) - 1 - torch.argmax(non_zero_mask.flip(dims=[1]), dim=1)

    # Gather the first and last non-zero elements in each row
    rows = torch.arange(T.size(0)).unsqueeze(1).to(T.device)
    first_non_zero_elements = T[rows, first_non_zero_idx.unsqueeze(1)].squeeze()
    last_non_zero_elements = T[rows, last_non_zero_idx.unsqueeze(1)].squeeze()

    # Calculate products
    product_first = torch.prod(first_non_zero_elements, dim=0)
    product_last = torch.prod(last_non_zero_elements, dim=0)

    # Calculate min and max
    min_val = torch.min(product_first, product_last)
    max_val = torch.max(product_first, product_last)

    return min_val, max_val

def global_min_max(T, t):
    # Split the tensor into sub-matrices with n rows each
    sub_matrices = T.chunk(t, dim=0)

    # Initialize global min and max values
    global_min = torch.tensor(float('0.')).to(T.device)
    global_max = torch.tensor(float('0.')).to(T.device)

    # Iterate over each sub-matrix using a comprehension
    for Tk in sub_matrices:
        min_val, max_val = min_max_product(Tk)
        global_min += min_val
        global_max += max_val

    return global_min, global_max

# Example usage
T = torch.tensor([[4, 8, 16, 0], [1, 2, 4, 8], [-60, -90, -120, 0], [1, 4/3, 5/3, 2]]).to(torch.float)
t = 2  # Number of sub-matrices
n = 2  # Rows in each sub-matrix
global_min, global_max = global_min_max(T, t)
print("Global Min:", global_min.item(), "Global Max:", global_max.item())


minmax = ibf_minmax_cpp.ibf_minmax(T.reshape(t, n, -1).cuda())
print(minmax)
minmax = ibf_minmax_cpp.quadrant_ibf_minmax(T.reshape(t, n, -1).cuda())
print(minmax)

