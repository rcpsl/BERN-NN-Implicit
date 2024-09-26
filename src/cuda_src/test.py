import torch
import ibf_minmax_cpp

nvars = 3
max_degree = 2

# # This example is from the slide deck
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


minmax = ibf_minmax_cpp.ibf_minmax(poly)
print(minmax)
assert minmax[0] == 0  # 0 * n + 0 * n
assert minmax[1] == 24 # 9 * 2 + 3 * 2
# minmax = ibf_minmax_cpp.ibf_minmax(poly)
# print(minmax)
# assert minmax[0] == 0  # 0 * n + 0 * n
# assert minmax[1] == 24 # 9 * 2 + 3 * 2

# poly = torch.tensor([
#     [
#         [-1, 1, 0],
#         [5, 6, 9],
#     ],
#     [
#         [-1, 1, 0],
#         [3, 3.5, 4]
#     ]
# ]).float().cuda()

# minmax = ibf_minmax_cpp.ibf_minmax(poly)
# print(minmax)
# assert minmax[0] == -13 # -1 * 9 + -1 * 4
# assert minmax[1] == 13  # 9 * 1 + 4 * 1


# # N terms, N variables, N powers.
# N = 8
# poly = torch.ones(N, N, N).cuda().float()
# minmax = ibf_minmax_cpp.ibf_minmax(poly)
# print(minmax)
# assert minmax[0] == N # each outer product returns 1, sum of N 1's is N
# assert minmax[1] == N # each outer product returns 1, sum of N 1's is N

# # N terms, N variables, N powers.
# N = 8
# poly = torch.ones(N, N, N).cuda().float()
# poly[0, 1, 0] = -1
# minmax = ibf_minmax_cpp.ibf_minmax(poly)
# print(minmax)
# assert minmax[0] == N - 2 # -1 + N-1 = N - 2 
# assert minmax[1] == N

# # N terms, N variables, N powers.
# N = 1597084
# poly = torch.ones(N, 4, 4).cuda().float()
# poly[-1, -1, -1] = -1
# minmax = ibf_minmax_cpp.ibf_minmax(poly)
# print(minmax)
# assert minmax[0] == N - 2 # -1 + N-1 = N - 2 
# assert minmax[1] == N


# # M terms, N variables, 2 powers.
# M = 5
# N = 10
# poly = torch.ones(M, N, 2).cuda().float()
# poly[:,:,0] = -1
# minmax = ibf_minmax_cpp.ibf_minmax(poly)
# print(minmax)
# assert minmax[0] == -M 
# assert minmax[1] == M

# 3597084 // 4 terms, 4 variables, 5 powers
M = 500
N = 7
P = 3
poly = torch.ones(M, N, P).cuda().float()
poly[-1,-1,-1] = -1
minmax = ibf_minmax_cpp.ibf_minmax(poly)
print(minmax)
assert minmax[0] == M - 2 
assert minmax[1] == M

# 3597084 // 4 terms, 4 variables, 5 powers
M = 5000000
N = 4
P = 5
poly = torch.ones(M, N, P).cuda().float()
poly[-1,-1,-1] = -1
minmax = ibf_minmax_cpp.ibf_minmax(poly)
print(minmax)
assert minmax[0] == M - 2 
assert minmax[1] == M

# 3597084 // 4 terms, 4 variables, 5 powers
M = 500000
N = 8
P = 6
poly = torch.ones(M, N, P).cuda().float()
poly[-1,-1,-1] = -1
minmax = ibf_minmax_cpp.ibf_minmax(poly)
print(minmax)
assert minmax[0] == M - 2 

# 3597084 // 4 terms, 4 variables, 5 powers
"""
M = 5000
N = 8
P = 10
poly = torch.ones(M, N, P).cuda().float()
poly[-1,-1,-1] = -1
minmax = ibf_minmax_cpp.ibf_minmax(poly)
print(minmax)
assert minmax[0] == M - 2 
"""
