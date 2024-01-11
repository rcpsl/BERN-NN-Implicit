import torch
import numpy as np
import time

# Assuming your extension is named 'row_wise_convolution_wrapper', and you
# have a function called 'row_convolution' within that extension.
import row_wise_convolution_wrapper

# Set device to CUDA to run on GPU
device = torch.device('cuda')

# Create some example data for T1 and T2
# Ensure the data type is float32, and they are on the GPU (device)

# T1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device, dtype=torch.float32)
# T2 = torch.tensor([[1, 2, 7], [0, 1, 6]], device=device, dtype=torch.float32)
T1 = torch.randn((1000000, 60), device=device, dtype=torch.float32)
T2 = torch.randn((1000000, 60), device=device, dtype=torch.float32)

start_time = time.time()
# Call the convolution function from your compiled extension
result = row_wise_convolution_wrapper.row_convolution(T1, T2)
print('time: ', time.time() - start_time)
# Print the result
print(result)

# res = T1
# C_diff = T2
# l_diff = torch.tensor([C_diff.size(1) - 1], device=device, dtype=torch.float32)
# start_time = time.time()
# result = [torch.nn.functional.conv1d(res[i, :].reshape(1, -1).reshape(1, 1, -1),      torch.flip(C_diff[i, :].reshape(1, -1), dims=(1,)).reshape(1, 1, -1), padding = torch.max(l_diff).int().item()   ).flatten()             for i in range(res.size(0) )]
# result = torch.stack(result)
# print('time: ', time.time() - start_time)

start_time = time.time()
print(np.convolve([1, 2, 3, 4], [1, 2, 7]))
print(np.convolve([5, 6, 7, 8], [0, 1, 6]))
print('time: ', time.time() - start_time)