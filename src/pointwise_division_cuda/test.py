import torch
import pointwise_div_extension
import time







# T1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 0.0]], device='cuda')
# T2 = torch.tensor([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]], device='cuda')
T1 = torch.rand((10000000, 160), device='cuda')
T2 = torch.rand((10000000, 160), device='cuda')

# print(T1)
# print(T2)

start_time = time.time()
result = pointwise_div_extension.pointwise_division(T1, T2)
end_time = time.time()
print("CUDA time: {}".format(end_time - start_time))
# print(result)


start_time = time.time()
result =  torch.where(T2 != 0, torch.div(T1, T2), 0.0)
end_time = time.time()
print("Python time: {}".format(end_time - start_time))
# print(result)

### compute sum of absolute differences
print(torch.sum(torch.abs(result - pointwise_div_extension.pointwise_division(T1, T2))))


