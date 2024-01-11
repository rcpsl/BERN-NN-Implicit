import torch
import repeat_terms_2_extension as rte
import time




def repeat_terms_2(TA, tA, n):

    TA_mod2 = [TA[i * n : ].clone() for i in  range(tA)]

    for i in range(len(TA_mod2)):
        chunk = TA_mod2[i]
        chunk[torch.arange(n, chunk.size(0), n)] *= 2

    TA_mod2 = torch.cat(TA_mod2, dim=0)

    return TA_mod2




# Example usage
device = torch.device('cuda')
n = 1500
tA = 100
# TA = torch.tensor([[1, 2, 3], [2, 5, 6]], device=device, dtype=torch.float32)
# TA = torch.tensor([[1, 2, 3], [2, 5, 6], [1, 2, 3], [2, 5, 6]], device=device, dtype=torch.float32)
TA = torch.rand((n, 10), device=device, dtype=torch.float32)
TA = TA.repeat(tA, 1)
n_columns = TA.size(1)


start_time = time.time()
result_cuda = rte.repeat_terms_2(TA, tA, n, n_columns)
end_time = time.time()
print("CUDA time: {}".format(end_time - start_time))

start_time = time.time()
result_python = repeat_terms_2(TA, tA, n)  # The original Python function
end_time = time.time()
print("Python time: {}".format(end_time - start_time))
# print(result_cuda)
# print(result_python)

### compute sum of absolute differences
print(torch.sum(torch.abs(result_cuda - result_python)))
































