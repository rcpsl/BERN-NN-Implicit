import torch
import ibf_dense_cpp

nvars = 3
max_degree = 2

# This example is from the slide deck
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


ebf = ibf_dense_cpp.ibf_dense(poly)
print(ebf)
