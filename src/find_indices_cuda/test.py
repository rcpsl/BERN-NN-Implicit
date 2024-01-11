import torch
import find_indices_extension
import time


def find_indices(degree):

    sizes = (degree + 1).tolist()
    mat = torch.ones(sizes, dtype=torch.float32)   
    indices = torch.nonzero(mat)

    return indices

# def find_indices(degree):
#     """
#     Generates indices for the given degree tensor using torch.meshgrid.
#     :param degree: The tensor of degrees.
#     :return: A tensor containing all indices.
#     """
#     # Generate a list of torch.arange tensors for each degree
#     ranges = [torch.arange(d + 1, device=degree.device) for d in degree]
    
#     # Use torch.meshgrid to generate the grid of indices
#     mesh = torch.meshgrid(*ranges)
    
#     # Flatten and stack to get the final indices tensor
#     indices = torch.stack([m.flatten() for m in mesh], dim=1)
    
#     return indices



degree = torch.tensor([2] * 19, dtype=torch.int32, device='cuda')
batch_size = 3 ** 15
# start_time = time.time()
# indices1 = find_indices_extension.find_indices(degree, batch_size)
# print('time', time.time() - start_time)
# print(indices1)

start_time = time.time()
indices2 = find_indices(degree)
print('time', time.time() - start_time)
# print(indices2)

### compute the percentage of zero elements in indices2
# print(torch.sum(indices2 == 0).item() / indices2.numel())
# print(torch.sum(indices2 == 1).item() / indices2.numel())
# print(torch.sum(indices2 == 2).item() / indices2.numel())

### compute the difference of the absolute between the two
# print(torch.sum(torch.abs(indices1 - indices2.cpu())))

