import torch
import generate_binom_coeffs_extension as gbce
import time


###################################################
### given a tensor of degrees L = tensor([L1,...,Ln]),
###   it outputs 2D 
### tensor([[(L1 choose 0),...,(L1 choose L1)],
###             .
###             .
###         [(Ln choose 0),...,(Ln choose Ln)]])
###################################################
def generate_binom_coeffs(L):
    device = L.device
    ranges = [torch.arange(i + 1) for i in L]
    N = L.reshape([len(L), 1])
    R = torch.nn.utils.rnn.pad_sequence(ranges, batch_first=True, padding_value = -1).to(device)
    A = torch.lgamma(N + 1)
    B = torch.lgamma(N - R + 1)
    C = torch.lgamma(R + 1)
    return torch.exp(A - B - C)


# Example usage
L = torch.tensor([3., 4., 5., 10., 11., 11., 3., 4., 5., 10., 11., 11., 3., 4., 5., 10., 11., 11., 3., 4., 5., 10., 11., 11.], device=torch.device('cuda'), dtype=torch.float32)

start_time = time.time()
result_cuda = gbce.generate_binom_coeffs(L)
end_time = time.time()
print("CUDA time: ", end_time - start_time)

start_time = time.time()
result_python = generate_binom_coeffs(L)
end_time = time.time()
print("Python time: ", end_time - start_time)

# print("CUDA result: ", result_cuda)
# print("Python result: ", result_python)

print("Difference: ", torch.sum(torch.abs(result_cuda - result_python)))
