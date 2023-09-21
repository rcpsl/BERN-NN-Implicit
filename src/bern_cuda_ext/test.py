import torch
import ibf_minmax_cpp

nvars = 3
max_degree = 2

#poly = torch.ones(nvars, max_degree).cuda()

poly = torch.tensor([
    [
        [4, 6, 9],
        [1, 2, 0],
    ],
    [
        [2, 2.5, 3],
        [1, 2,   0]
    ]
]).float().cuda()


# minmax should be a tensor of two elements.
minmax = ibf_minmax_cpp.ibf_minmax(poly)
print(minmax)

poly = torch.tensor([
    [
        [-1, 1, 0],
        [5, 6, 9],
    ],
    [
        [-1, 1, 0],
        [3, 3.5, 4]
    ]
]).float().cuda()


# minmax should be a tensor of two elements.
minmax = ibf_minmax_cpp.ibf_minmax(poly)
print(minmax)
