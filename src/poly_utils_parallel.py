import torch
from utils import * 


###################################################
### given a tensor of degrees L = tensor([L1,...,Ln]),
###   it outputs 2D 
### tensor([[(L1 choose 0),...,(L1 choose L1)],
###             .
###             .
###         [(Ln choose 0),...,(Ln choose Ln)]])
###################################################
def generate_binom_coeffs(L, device='cuda:0'):
    ranges = [torch.arange(i + 1, device=device) for i in L]
    N = L.reshape([len(L), 1]).to(device)
    R = torch.nn.utils.rnn.pad_sequence(ranges, batch_first=True, padding_value=-1).to(device)
    A = torch.lgamma(N + 1)
    B = torch.lgamma(N - R + 1)
    C = torch.lgamma(R + 1)
    return torch.exp(A - B - C)









def split_and_select(pA, tA, I, device):
    # Move tensors to the specified device
    pA = pA.to(device)
    
    # Calculate the number of rows in each subtensor
    rows_per_subtensor = pA.size(0) // tA

    # Reshape the tensor into tA subtensors
    subtensors = pA.split(rows_per_subtensor)

    # Stack the subtensors along a new dimension to select rows
    stacked_subtensors = torch.stack(subtensors).to(device)
    
    # Use tensor indexing to select rows based on indices
    selected_rows = stacked_subtensors[:, I].to(device)
    
    # Reshape the result to a 2D tensor
    pAmod = selected_rows.view(-1, selected_rows.size(-1))

    # Find the row indices where pAmod matches pA
    idx_list = torch.where(torch.all(pA.unsqueeze(1) == pAmod, dim=2))[0]

    return pAmod, idx_list



###################################################
### given a 2D tensor polynomial pA and a constant 
### c, it performs the addition pA + c:
### c = tensor([c c ...],
###               [1 1 ...],
###                   .
###               [1 1 ...])
### n: number of variables
### tA: number of terms in pA
### degree_A: degree of pA
###################################################
def add_with_constant(n, pA, const, device):
    # Ensure the tensors are on the specified device
    pA = pA.to(device)
    const = const.to(device)
    
    ### replace the non-zero elements for the first n rows of pA with 1s
    c_tensor = torch.where(pA[0 : n, :] != 0.0, 1.0, 0.0).to(device)
    ### multiply the first row of c_tensor with const
    c_tensor[0, :] *= const
    ### concatenate c_tensor below pA along the rows
    return torch.cat((pA, c_tensor), 0)




###################################################
### given a 2D tensor polynomial pA and a constant 
### c, it performs the multiplication c * pA:
### multiply the row with c and jump n rows
### n: number of variables
### tA: number of terms in pA
### degree_A: degree of pA
###################################################
def mult_with_constant(n, pA, c, device):
    """
    Multiplies every row of the tensor by 'c', skipping 'n' rows in between each multiplication.
    
    Parameters:
        n (int): The number of rows to skip.
        pA (torch.Tensor): The input 2D tensor.
        c (float): The constant to multiply.
        device (str): The device to perform the operation on.

    Returns:
        torch.Tensor: The resulting tensor after the operation.
    """
    # Move tensors to the specified device
    pA = pA.to(device)
    c = c.to(device) if isinstance(c, torch.Tensor) else c

    # Ensure the input is a 2D tensor
    assert pA.dim() == 2, "Input needs to be a 2D tensor"

    # Step 1: Create a multiplier tensor
    num_rows = pA.shape[0]  # Number of rows in the original tensor
    num_row_ones = torch.ones(num_rows, device=device)
    vect = torch.arange(num_rows, device=device)
    multiplier = torch.where(vect % n == 0, c * num_row_ones, num_row_ones)
    multiplier = multiplier.view(-1, 1)  # Ensure it's the correct shape for broadcasting

    # Step 2: Multiply
    result = pA * multiplier  # Element-wise multiplication with broadcasting

    return result







def degree_elevation(pA, n, tA, degree, new_degree, device):
    pA = pA.to(device)
    degree = degree.to(device)
    new_degree = new_degree.to(device)
    # print('pA', pA)
    # print('degree', degree)
    # print('new_degree', new_degree)

    l_diff = new_degree - degree
    I_non_zero = torch.nonzero(l_diff).reshape(-1).to(device)
    degree_non_zero = degree[I_non_zero]
    l_diff_non_zero = l_diff[I_non_zero]
    new_degree_non_zero = new_degree[I_non_zero]

    if len(I_non_zero) == n:
        pAmod = pA
    else:
        pAmod, idx_list = split_and_select(pA, tA, I_non_zero, device)

    # Here, ensure that the generate_binom_coeffs function is also modified to accept a device parameter
    C_degree = generate_binom_coeffs(degree_non_zero, device)
    C_degree = C_degree.repeat(tA, 1)

    C_diff = generate_binom_coeffs(l_diff_non_zero, device)
    C_diff = C_diff.repeat(tA, 1)

    C_new_degree = generate_binom_coeffs(new_degree_non_zero, device)
    C_new_degree = C_new_degree.repeat(tA, 1)

    res = torch.mul(pAmod, C_degree)

    # Perform the convolution for each row

    result = [torch.nn.functional.conv1d(res[i, :].reshape(1, -1).reshape(1, 1, -1),      torch.flip(C_diff[i, :].reshape(1, -1), dims=(1,)).reshape(1, 1, -1), padding = torch.max(l_diff).int().item()   ).flatten()             for i in range(res.size(0) )]
    result = torch.stack(result)
    
    # print('result', result)
    # print('C_new_degree', C_new_degree)
    result = torch.where(C_new_degree != 0, torch.div(result, C_new_degree), 0.0)

    ### pad result tensor with 0 columns of number max(new_degree) - max(degree) 
    pA_elevated = torch.nn.functional.pad(pA, (0, torch.max(l_diff).int().item()))

    if len(I_non_zero) == n:
        pA_elevated = result
    else:
        pA_elevated[idx_list, :] = result
    
    # ### remove zero rows from pA_elevated
    # pA_elevated = remove_zero_rows(pA_elevated)
    return pA_elevated




def process_tensor(T, n, device):
    T = T.to(device)  # Move the input tensor to the specified device

    # Step 1: Chunking
    chunks = torch.chunk(T, T.size(0) // n, dim=0)
    
    # Step 2: Check equality everywhere except the first row
    mask = torch.ones_like(chunks[0], device=device)  # Create the mask on the specified device
    mask[0, :] = 0

    # Use broadcasting to compare chunks while ignoring the first row
    chunks_broad = torch.stack(chunks).to(device)  # Ensure the stacked chunks are on the specified device
    equality = (chunks_broad.unsqueeze(1) * mask == chunks_broad.unsqueeze(0) * mask).all(dim=-1).all(dim=-1)
    
    # Identify chunks that need the first row summed
    rows_1_to_sum, rows_2_to_sum = equality.triu(diagonal=1).nonzero(as_tuple=True)

    # Create an array to track which chunks to keep, on the specified device
    to_keep = torch.ones(len(chunks), dtype=torch.bool, device=device)

    # Apply the summations for chunks that need it
    for i1, i2 in zip(rows_1_to_sum, rows_2_to_sum):
        chunks_broad[i1, 0, :] += chunks_broad[i2, 0, :]
        to_keep[i2] = False  # Mark the second tensor as not to keep
    
    # Step 3: Concatenate the chunks to get the resultant tensor
    chunks_to_concat = [chunk for i, chunk in enumerate(chunks_broad) if to_keep[i]]
    result = torch.cat(chunks_to_concat, dim=0)
    return result







###################################################
### performs summation between 2 2D tensor
###  polynomials pA and pB.
### n: number of variables
### tA: number of terms in pA
### degree_A: degree of pA
### tB: number of terms in pB
### degree_B: degree of pB
###################################################    

def sum_2_polys(n, pA, tA, degree_A, pB, tB, degree_B, device='cuda:0'):
    # Move tensors to the specified device
    pA = pA.to(device)
    pB = pB.to(device)
    degree_A = degree_A.to(device)
    degree_B = degree_B.to(device)

    ### if degree_A is all zeros then return pB
    if torch.all(degree_A == 0):
        return pB
    ### if degree_B is all zeros then return pA
    elif torch.all(degree_B == 0):
        return pA

    ### concatenate pA and pB 
    if torch.all(degree_A == degree_B):
        T = torch.cat((pA, pB), dim=0)
        return process_tensor(T, n, device)
    
    ### degree elevate pA concatenate it with pB
    elif torch.all(torch.ge(degree_B, degree_A)):
        pA_elevated = degree_elevation(pA, n, tA, degree_A, degree_B, device)
        T = torch.cat((pA_elevated, pB), dim=0)
        return process_tensor(T, n, device)

    ### degree elevate pB concatenate it with pA
    elif torch.all(torch.ge(degree_A, degree_B)):
        pB_elevated = degree_elevation(pB, n, tB, degree_B, degree_A, device)
        # print('degree_A', degree_A)
        # print('degree_B', degree_B)
        # print('pA', pA)
        # print('pB', pB)
        # print('pB_elevated', pB_elevated)
        T = torch.cat((pA, pB_elevated), dim=0)
        return process_tensor(T, n, device)

    ### degree elevate pA and pB concatenate them
    else:
        degree_max = torch.max(degree_A, degree_B)
        # print('pA', pA)
        # print('degree_A', degree_A)
        # print('degree_B', degree_B)
        # print('degree_max', degree_max)
        pA_elevated = degree_elevation(pA, n, tA, degree_A, degree_max, device)
        pB_elevated = degree_elevation(pB, n, tB, degree_B, degree_max, device)
        T = torch.cat((pA_elevated, pB_elevated), dim=0)
        return process_tensor(T, n, device)



###################################################
### performs power 2 of a 2D polynomial tensor TA  \
### tA: number of terms in pA
### degree_A: degree of pA
###################################################
def poly_pow_2(n_vars, TA, tA, degree_A, device):
    # Move input tensors to the specified device
    TA = TA.to(device)
    degree_A = degree_A.to(device)
    
    # Operations that create new tensors should also specify the device
    # Modify the repeat_terms function to accept a device parameter
    TA_mod1 = repeat_terms_device(TA, tA, n_vars, device)
    
    # Operations that create new tensors should also specify the device
    TA_mod2 = [TA[i * n_vars:].clone().to(device) for i in range(tA)]
    for i in range(len(TA_mod2)):
        chunk = TA_mod2[i]
        chunk[torch.arange(n_vars, chunk.size(0), n_vars, device=device)] *= 2
    TA_mod2 = torch.cat(TA_mod2, dim=0)
    
    # Modify the generate_binom_coeffs function to accept a device parameter
    C_A = generate_binom_coeffs(degree_A, device)
    C_A = C_A.repeat(int((tA ** 2 + tA) / 2), 1)

    degree_mul = 2 * degree_A
    C_AmulA = generate_binom_coeffs(degree_mul, device)
    C_AmulA = C_AmulA.repeat(int((tA ** 2 + tA) / 2), 1)

    T_C_A = torch.mul(TA_mod1, C_A)
    T_C_B = torch.mul(TA_mod2, C_A)
    
    n_rows, m = T_C_A.shape
    T_C_A = T_C_A.view((1, n_rows, m))
    T_C_B = T_C_B.view((n_rows, 1, m))
    T_C_B_flipped = torch.flip(T_C_B, dims=(2,))
    del T_C_B

    res = torch.nn.functional.conv1d(T_C_A, T_C_B_flipped, padding  = m - 1, groups = n_rows).flatten(1)
    del T_C_A
    del T_C_B_flipped
    res = res.reshape(-1, 2 * m - 1)
    
    res /= C_AmulA
    torch.nan_to_num(res, nan=0.0, out=res)
    
    return res


###################################################
### given a polynomial pA and quadratic polynomial
### q(x) = a x^2 + b x + c, computes:  
### p_quad = q(pA) = a pA^2 + b pA + c  
### n: number of variables
### tA: number of terms in pA
### degree_A: degree of pA
### coeffs = [a, b, c] : coeffs of quadratic poly q(x)
###################################################
def quad_of_poly(n, pA, tA, degree_A, coeffs, device='cuda:0'):
    # Move coefficients and other tensors to the specified device
    coeffs = coeffs.to(device)
    pA = pA.to(device)
    degree_A = degree_A.to(device)

    if torch.abs(coeffs[0]) != 0.0:
        # Compute pA ** 2, ensuring poly_pow_2 accepts and uses the device parameter
        res = poly_pow_2(n, pA, tA, degree_A, device)
        # Compute a * pA ** 2, ensuring mult_with_constant accepts and uses the device parameter
        term1 = mult_with_constant(n, res, coeffs[0], device)
        # Compute b * pA, ensuring mult_with_constant accepts and uses the device parameter
        term2 = mult_with_constant(n, pA, coeffs[1], device)
        # Sum term1 + term2: a * pA ** 2 + b * pA, ensuring sum_2_polys accepts and uses the device parameter
        t_term1 = term1.size(0) // n
        degree_term1 = 2 * degree_A
        term = sum_2_polys(n, term1, t_term1, degree_term1, term2, tA, degree_A, device)
    else:
        # Compute b * pA, ensuring mult_with_constant accepts and uses the device parameter
        term = mult_with_constant(n, pA, coeffs[1], device)

    # Sum term + c if c != 0; a * pA ** 2 + b * pA + c, ensuring add_with_constant accepts and uses the device parameter
    if torch.abs(coeffs[2]) != 0.0:
        return add_with_constant(n, term, coeffs[2], device)
    else:
        return term














# if __name__ == "__main__":


    # ### 1) test degree_elevation function
    # pA = torch.tensor([[1.], [1.], [1.]])
    # n = 3
    # tA = 1
    # degree = torch.tensor([0, 0, 0])
    # new_degree = torch.tensor([1, 1, 2])
    # res = degree_elevation(pA, n, tA, degree, new_degree)
    # print(res)

    # ### 2) test sum_2_polys function
    # n = 2
    # pA = torch.tensor([[1., 2., 4., 8.], [4., 8., 16., 0]])
    # tA = 1
    # degree_A = torch.tensor([3, 2])
    # pB = torch.tensor([[1., 2.], [2., 4.]])
    # tB = 1
    # degree_B = torch.tensor([1, 1])
    # res = sum_2_polys(n, pA, tA, degree_A, pB, tB, degree_B)
    # print(res)

    # ### 3) test mult_2_polys function
    # n = 2
    # pA = torch.tensor([[1., 2.], [4., 8.]])
    # tA = 1
    # degree_A = torch.tensor([1, 1])
    # pB = torch.tensor([[1., 2.], [4., 8.]])
    # tB = 1
    # degree_B = torch.tensor([1, 1])
    # res = mult_2_polys(n, pA, tA, degree_A,  pB, tB, degree_B)
    # print(res)

    # ### 4) test mult_with_constant function
    # n = 2
    # pA = torch.tensor([[1., 2.], [4., 8.], [1., 2.], [4., 8.], [1., 2.], [4., 8.]])
    # tA = 3
    # degree_A = torch.tensor([1, 1])
    # c = 5
    # res = mult_with_constant(n, pA, tA, degree_A, c)
    # print(res)

    ### 5) test add_with_constant function
    # n = 2
    # pA = torch.tensor([[1., 2.], [4., 8.]])
    # c = 2
    # res = add_with_constant(n, pA, c)
    # print(res)

    ### 6) test quad_of_poly function
    # n = 2
    # pA = torch.tensor([[1., 2., 3], [4., 8., 3]])
    # tA = 1
    # degree_A = torch.tensor([2, 2])
    # coeffs = torch.tensor([1, 2, 3])
    # res = quad_of_poly(n, pA, tA, degree_A, coeffs)
    # print(res)






    




    


    