import torch
from utils import * 

import sys
import row_wise_conv_cpp


###################################################
### given a tensor of degrees L = tensor([L1,...,Ln]),
###   it outputs 2D 
### tensor([[(L1 choose 0),...,(L1 choose L1)],
###             .
###             .
###         [(Ln choose 0),...,(Ln choose Ln)]])
###################################################
def generate_binom_coeffs(L):
    ranges = [torch.arange(i + 1) for i in L]
    N = L.reshape([len(L), 1])
    R = torch.nn.utils.rnn.pad_sequence(ranges, batch_first=True, padding_value = -1) 
    A = torch.lgamma(N + 1)
    B = torch.lgamma(N - R + 1)
    C = torch.lgamma(R + 1)
    return torch.exp(A - B - C)


def split_and_select(pA, tA, I):
    # Calculate the number of rows in each subtensor
    rows_per_subtensor = pA.size(0) // tA

    # Reshape the tensor into tA subtensors
    subtensors = pA.split(rows_per_subtensor)

    # Stack the subtensors along a new dimension to select rows
    stacked_subtensors = torch.stack(subtensors)
    
    # Use tensor indexing to select rows based on indices
    selected_rows = stacked_subtensors[:, I]
    # Reshape the result to a 2D tensor
    pAmod = selected_rows.view(-1, selected_rows.size(-1))

    # print(pAmod)
    # print(pA)

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
def add_with_constant(n, pA, const):
    ### replace the non-zero elements for the first n rows of pA with 1s
    c_tensor = torch.where(pA[0 : n, :] != 0.0, 1.0, 0.0)
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
def mult_with_constant(n, pA, c):
    """
    Multiplies every row of the tensor by 'c', skipping 'n' rows in between each multiplication.
    
    Parameters:
        tensor (torch.Tensor): The input 2D tensor.
        c (float): The constant to multiply.
        n (int): The number of rows to skip.

    Returns:
        torch.Tensor: The resulting tensor after the operation.
    """
    device = pA.device
    # Ensure the input is a 2D tensor
    assert pA.dim() == 2, "Input needs to be a 2D tensor"

    # Step 1: Create a multiplier tensor

    # Number of rows in the original tensor
    num_rows = pA.shape[0]

    # Create a vector with 'c' at positions where rows need to be multiplied, and '1' elsewhere
    # 'floor_divide' calculates the integer division; positions to be multiplied are those where (index // n) is even
    # print('device', device)
    # print('c.device', c.device)
    
    num_row_ones = torch.ones(num_rows).to(device)
    vect = torch.arange(num_rows).to(device)
    # print('num_row_ones.device', num_row_ones.device)
    multiplier = torch.where(vect % n == 0, c * num_row_ones, num_row_ones)

    # Ensure it's the correct shape for broadcasting
    multiplier = multiplier.view(-1, 1)

    # Step 2: Multiply

    # Element-wise multiplication; thanks to broadcasting, 'multiplier' is virtually expanded to match 'tensor's shape
    result = pA * multiplier

    return result






def degree_elevation(pA, n, tA, degree, new_degree):
    # print('pA', pA)
    # print('degree', degree)
    # print('new_degree', new_degree)
    l_diff = new_degree - degree
    I_non_zero = torch.nonzero(l_diff).reshape(-1)
    degree_non_zero = degree[I_non_zero]
    # print(I_non_zero)
    l_diff_non_zero = l_diff[I_non_zero]
    new_degree_non_zero = new_degree[I_non_zero]
    ### no split_and_select from pA if we degree_elevate all degree and split_and_select otherwise
    if len(I_non_zero) == n:
        pAmod = pA
    else:
        pAmod, idx_list = split_and_select(pA, tA, I_non_zero)
    ### generate binomial coefficients and repeated it tA times
    C_degree = generate_binom_coeffs(degree_non_zero)
    C_degree = C_degree.repeat(tA, 1)

    C_diff = generate_binom_coeffs(l_diff_non_zero)
    C_diff = C_diff.repeat(tA, 1)

    C_new_degree = generate_binom_coeffs(new_degree_non_zero)
    C_new_degree = C_new_degree.repeat(tA, 1)
    
    # print(C_degree)
    # print(C_diff)
    # print(C_new_degree)
    
    ### point-wise multiplication between pAmod * C_degree
    res = torch.mul(pAmod, C_degree)
    # print(res)
    # print(res.size(0))
    ### perform row-wise convolution between res and C_diff
    # print('res', res)
    # print('C_diff', C_diff)
    # print('l_diff', l_diff)
    # print('new_degree', new_degree)

    ### compute the convolution of res and C_diff 
    result = row_wise_conv_cpp.row_convolution(res, C_diff)
    # print(result)


    # print('result', result)
    # print('C_new_degree', C_new_degree)
    ### perform row-wise division between res_conv and C_new_degree for non-zero elements of C_new_degree
    result =  torch.where(C_new_degree != 0, torch.div(result, C_new_degree), 0.0)
    
    ### pad result tensor with 0 columns of number max(new_degree) - max(degree) 
    pA_elevated = torch.nn.functional.pad(pA, (0, torch.max(l_diff).int().item()))

    ### replace the rows of pA_elevated with the rows of result given by the row indices idx_list

    # print(pA_elevated)
    # print(idx_list)
    # print(result)
    # print('pA_elevated', pA_elevated)
    # print('idx_list', idx_list)
    # print('result', result)
    if len(I_non_zero) == n:
        pA_elevated = result
    else:
        pA_elevated[idx_list, :] = result
    
    # ### remove zero rows from pA_elevated
    # pA_elevated = remove_zero_rows(pA_elevated)
    return pA_elevated




# # ###################################################
# # ### performs summation between 2 2D tensor
# # ###  polynomials pA and pB of same degrees.
# # ### n: number of variables
# # ### tA: number of terms in pA
# # ### tB: number of terms in pB
# # ###################################################
# def sum_2_polys_same_degree(n, pA, tA, pB, tB):
    

#     # ### if pA.size() == pB.size() then sum every n rows of pA and pB
#     # if pA.size() == pB.size():
        
#     ### sum only the first n rows of pA and pB if pA and pB are the same execpt for the first n rows
#     res = [torch.cat(((pA[i * n , :] + pB[j * n, :]).unsqueeze(0), pA[i * n + 1 : (i + 1) * n, :])) for j in range(tB)   for i in range(tA) if torch.equal(pA[i * n + 1 : (i + 1) * n, :], pB[j * n + 1 : (j + 1) * n, :])]
#     ### concatenate res along the rows and return it
#     # print('res', res)
#     if len(res) == 0:
#         return torch.cat((pA, pB), 0)
#     return torch.cat(res, dim=0)


def sum_2_polys_same_degree(n, T1, T2):

    # Step 1: Chunking
    T1_chunks = torch.chunk(T1, T1.size(0) // n)
    T2_chunks = torch.chunk(T2, T2.size(0) // n)

    # Step 2: Checking Sub-tensors for all cross-product pairs
    mask = torch.ones_like(T1_chunks[0])
    mask[0, :] = 0
    
    equal_except_first_row = [
        (chunk1 * mask == chunk2 * mask).all()
        for chunk1 in T1_chunks
        for chunk2 in T2_chunks
    ]

    # Process chunks based on the equality checks
    resultant_chunks = []
    for chunk1 in T1_chunks:
        i = 0
        for chunk2 in T2_chunks:
            equal = (chunk1 * mask == chunk2 * mask).all()
            print('chunk1 * mask', chunk1 * mask)
            print('chunk2 * mask', chunk2 * mask)
            print('equal', equal)
            if equal:
                # print('chunk1', chunk1)
                # print('chunk2', chunk2)
                summed_chunk = chunk1.clone()
                summed_chunk[0, :] += chunk2[0, :]
                # print('summed_chunk', summed_chunk)
                resultant_chunks.append(summed_chunk)
            else:
                # print('chunk1', chunk1)
                # print('chunk2', chunk2)
                resultant_chunks.append(chunk2)
                i += 1

        if i == (len(T2_chunks) - 1):
            resultant_chunks.append(chunk1)


    # Step 3: Return the resultant big tensor by concatenating the resultant chunk tensors
    result_tensor = torch.cat(resultant_chunks, dim=0)

    return result_tensor


# def process_tensor(T, n):
#     # Step 1: Chunking
#     chunks = torch.chunk(T, T.size(0) // n, dim=0)
#     # print('chunks', chunks)
#     # Step 2: Check equality everywhere except the first row
#     mask = mask = torch.ones_like(chunks[0])
#     mask[0, :] = 0
    
#     # Create an equality tensor
#     # This tensor will have a shape of (N/n, N/n) where entry (i, j) is True if chunk i and chunk j are equal (ignoring first row)
#     equality_tensor = torch.stack([((chunk * mask) == (other_chunk * mask)).all() for chunk in chunks for other_chunk in chunks]).view(len(chunks), len(chunks))

#     # Create a mask for the chunks that need the first row summed
#     sum_mask = equality_tensor.triu(diagonal=1)  # We use triu to avoid double-counting and self-counting


#     # Apply the summations for chunks that need it

#     rows_1_to_sum = sum_mask.nonzero(as_tuple=True)[0]  # Get the chunk indices that need to be summed
#     rows_2_to_sum = sum_mask.nonzero(as_tuple=True)[1]  # Get the chunk indices that need to be summed

#     result_tensor = []
#     for index1, index2 in zip(rows_1_to_sum, rows_2_to_sum):
#         res = chunks[index1] * mask + chunks[index2] * mask
#         result_tensor.append(res)


        
    
#     # Step 3: Concatenate the chunks to get the resultant tensor
#     result = torch.cat(chunks, dim=0)
#     return result


def process_tensor(T, n):
    # Step 1: Chunking
    chunks = torch.chunk(T, T.size(0) // n, dim=0)
    
    # Step 2: Check equality everywhere except the first row
    mask = torch.ones_like(chunks[0])
    mask[0, :] = 0

    # Use broadcasting to compare chunks while ignoring the first row
    chunks_broad = torch.stack(chunks)
    equality = (chunks_broad.unsqueeze(1) * mask == chunks_broad.unsqueeze(0) * mask).all(dim=-1).all(dim=-1)
    
    # Identify chunks that need the first row summed
    rows_1_to_sum, rows_2_to_sum = equality.triu(diagonal=1).nonzero(as_tuple=True)

    # Create an array to track which chunks to keep
    to_keep = torch.ones(len(chunks), dtype=torch.bool)

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

def sum_2_polys(n, pA, tA, degree_A, pB, tB, degree_B):
    # print('degree_A_beforedegree_A_beforedegree_A_beforedegree_A_beforedegree_A_before', degree_A)
    # print('degree_B_beforedegree_B_beforedegree_B_beforedegree_B_beforedegree_B_before', degree_B)
    
    ### if degree_A is all zeros then return pB
    if torch.all(degree_A == 0):
        return pB
    ### if degree_B is all zeros then return pA
    elif torch.all(degree_B == 0):
        return pA

    ### concatenate pA and pB 
    if torch.all(degree_A == degree_B):
        # print('tA', tA)
        # # print('tB', tB)
        # print('pA[0:10, :]', pA[0:10, :])
        # print('pB[0:10, :]', pB[0:10, :])
        # print('pA.size()', pA.size())
        # print('pB.size()', pB.size())
        # print('pA', pA)
        # print('pB', pB)
        # print('degree_A', degree_A)
        # print('degree_B', degree_B)
        return torch.cat((pA, pB), 0)
        # return sum_2_polys_same_degree(n, pA, tA, pB, tB)
        # return sum_2_polys_same_degree(n, pA, pB)
        T = torch.cat((pA, pB), dim=0)
        return process_tensor(T, n)
    


    ### degree elevate pA concatenate it with pB
    elif torch.all(torch.ge(degree_B, degree_A) ):
        pA_elevated = degree_elevation(pA, n, tA, degree_A, degree_B)
        return torch.cat((pA_elevated, pB), 0)
        T = torch.cat((pA_elevated, pB), dim=0)
        return process_tensor(T, n)

    ### degree elevate pB concatenate it with pA
    elif torch.all(torch.ge(degree_A, degree_B) ):
        pB_elevated = degree_elevation(pB, n, tB, degree_B, degree_A)
        return torch.cat((pA, pB_elevated), 0)
        # print('degree_A', degree_A)
        # print('degree_B', degree_B)
        # print('pA', pA)
        # print('pB', pB)
        # print('pB_elevated', pB_elevated)
        T = torch.cat((pA, pB_elevated), dim=0)
        return process_tensor(T, n)


    ### degree elevate pA and pB concatenate it with pB
    else:
        degree_max = torch.max(degree_A, degree_B)
        # print('pA', pA)
        # print('n', n)
        # print('tA', tA)
        # print('degree_A', degree_A)
        # print('degree_B', degree_B)
        # print('degree_max', degree_max)
        pA_elevated = degree_elevation(pA, n, tA, degree_A, degree_max)
        pB_elevated = degree_elevation(pB, n, tB, degree_B, degree_max)
        return torch.cat((pA_elevated, pB_elevated), 0)
        T = torch.cat((pA_elevated, pB_elevated), dim=0)
        return process_tensor(T, n)


###################################################
### performs multiplicaion between 2 2D terms  
###  tensors TA and TB
###################################################
def mult_2_terms(n, TA, C_A, TB, C_B, C_AmulB):

    ### perform multpilcation point-wise between TA and CA; TB and CB
    T_C_A = torch.mul(TA, C_A)
    T_C_B = torch.mul(TB, C_B)

    # ### 1) perform row-wise convolution between T_C_A and T_C_B   using for loop
    # res = [torch.nn.functional.conv1d(T_C_A[i, :].reshape(1, -1).reshape(1, 1, -1),      torch.flip(T_C_B[i, :].reshape(1, -1), dims=(1,)).reshape(1, 1, -1), padding = T_C_B.size(1) - 1   ).flatten()             for i in range(n)]
    # ### stack the rows
    # res = torch.stack(res) 

    ### 2) perform row-wise convolution between T_C_A and T_C_B   without using for loop
    # Assuming T_C_A and T_C_B are both 2D tensors of shape (n, m)
    n, m = T_C_A.shape
    # Reshape T_C_A and T_C_B to have an additional channel dimension
    T_C_A = T_C_A.unsqueeze(1)  # Shape: (n, 1, m)
    T_C_B = T_C_B.unsqueeze(1)  # Shape: (n, 1, m)
    # Flip T_C_B
    T_C_B_flipped = torch.flip(T_C_B, dims=(2,))  # Shape: (n, 1, m)
    # Perform convolution
    res = torch.nn.functional.conv1d(T_C_A, T_C_B_flipped, padding=m - 1).flatten(1)  # Shape: (n, m)
    res = res.reshape(-1, 2 * m - 1)
    res = res[torch.arange(0, res.size(0), step = n + 1)]
    
    ### perform row-wise division between res and C_AmulB for non-zero elements of C_AmulB
    res =  torch.where(C_AmulB != 0, torch.div(res, C_AmulB), 0.0)

    return res


###################################################
### performs multiplicaion between 2 2D tensor
###  polynomials pA and pB.
### n: number of variables
### tA: number of terms in pA
### degree_A: degree of pA
### tB: number of terms in pB
### degree_B: degree of pB
###################################################

def mult_2_polys(n, pA, tA, degree_A,  pB, tB, degree_B):

    ### compute 2D binomial coefficients for degree_A and degree_B
    C_A = generate_binom_coeffs(degree_A)
    C_B = generate_binom_coeffs(degree_B)

    ### compute 2D binomial coefficients for degree_mul = degree_A + degree_B
    degree_mul = degree_A + degree_B
    C_AmulB = generate_binom_coeffs(degree_mul)
    # print('22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222')
    ### perform multiplication between each term of pA to all the terms of pB
    # print('pAsize', pA.size())
    res = [mult_2_terms(n, pA[i * n : (i + 1) * n, :], C_A, pB[j * n : (j + 1) * n, :], C_B, C_AmulB) for j in range(tB)   for i in range(tA)]
    # print('22222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222')
    
    # print(res)
    ### stack the rows and reshape the final result to (number_of rows, torch.max(degree_mul) + 1)
    res = torch.stack(res).reshape(-1, torch.max(degree_mul).int().item() + 1)

    return res


###################################################
### performs power 2 of a 2D polynomial tensor TA  \
### tA: number of terms in pA
### degree_A: degree of pA
###################################################
def poly_pow_2(n_vars, TA, tA, degree_A):
    # print('tA', tA)
    ### create TA_mod1 = repeat every term in TA tA - index times along the first dimension
    TA_mod1 = repeat_terms(TA, tA, n_vars)
    # print('TA_mod1_size', TA_mod1.size())
    ### create TA_mod2 = reapeted TA tA times along the first dimension
    TA_mod2 = [TA[i * n_vars:].clone() for i in  range(tA)]
    for i in range(len(TA_mod2)):
        chunk = TA_mod2[i]
        chunk[torch.arange(n_vars, chunk.size(0), n_vars)] *= 2
        #TA_mod2[i][n_vars:chunk.size(0):n_vars] *= 2 #[torch.arange(n_vars, chunk.size(0), n_vars)] *= 2
    # print('TA_mod2', TA_mod2)
    TA_mod2 = torch.cat(TA_mod2, dim=0)
    # print('TA_mod2_size', TA_mod2.size())
    ### compute 2D binomial coefficients for degree_A 
    C_A = generate_binom_coeffs(degree_A)

    ### repeat C_A (tA ** 2 + tA) / 2 times along the first dimension
    C_A = C_A.repeat(int((tA ** 2 + tA) / 2), 1)
    # print('C_A_size', C_A.size())

    ### compute 2D binomial coefficients for degree_mul = 2 * degree_A 
    degree_mul = 2 * degree_A 
    C_AmulA = generate_binom_coeffs(degree_mul)
    ### repeat C_AmulA (tA ** 2 + tA) / 2 times along the first dimension
    C_AmulA = C_AmulA.repeat(int((tA ** 2 + tA) / 2), 1)
    # print('C_AmulA_size', C_AmulA.size())

    ### perform multpilcation point-wise between TA_mod1 and CA; TA_mod2 and CA
    # print(TA_mod1.shape)
    # print(C_A.shape)
    T_C_A = torch.mul(TA_mod1, C_A)
    T_C_B = torch.mul(TA_mod2, C_A)
    
    # # Assuming T_C_A and T_C_B are both 2D tensors of shape (n, m)
    n_rows, m = T_C_A.shape
    # Perform convolution
    res = row_wise_conv_cpp.row_convolution(T_C_A, T_C_B)
    del T_C_A
    del T_C_B
    res = res.reshape(-1, 2 * m - 1)
    # print(res)
        
    ### perform row-wise division between res and C_AmulA for non-zero elements of C_AmulA
    # print(res.shape)
    # print(C_AmulA.shape)
    # res =  torch.where(C_AmulA != 0, torch.div(res, C_AmulA), torch.tensor(0.))
    #print(res)
    #print(C_AmulA)
    res /= C_AmulA
    ### check where is Nan and replace it with 0
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
def quad_of_poly(n, pA, tA, degree_A, coeffs):

    
    # print(pA)
    # print(degree_A)
    # print(tA)
    # print(pA.size(0) // n)
    # res = mult_2_polys(n, pA, tA, degree_A,  pA, tA, degree_A)

    if torch.abs(coeffs[0]) != 0.0:
        # print('coeffs[0]', coeffs[0])
        # print('coeffs[1]', coeffs[1])
        ### compute pA ** 2
        res = poly_pow_2(n, pA, tA, degree_A)
        ### compute a * pA ** 2 
        term1 = mult_with_constant(n, res, coeffs[0])
        ### compute b * pA 
        term2 = mult_with_constant(n, pA, coeffs[1])
        ### sum term1 + term2: a * pA ** 2 + b * pA 
        t_term1 = term1.size(0) // n
        degree_term1 = 2 * degree_A
        term = sum_2_polys(n, term1, t_term1, degree_term1, term2, tA, degree_A)
    else:
        ### compute b * pA 
        # print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', coeffs[1])
        # print('coeffs[1]', coeffs[1])
        # print('pApApApApApApApApApApApApApApApApApApApApApApApApApApApA', pA)
        term = mult_with_constant(n, pA, coeffs[1])
        # print('termtermtermtermtermtermtermtermtermtermtermtermtermterm', term)

    ### sum term + c if c != 0; a * pA ** 2 + b * pA + c
    if torch.abs(coeffs[2]) != 0.0:
        # print('coeffs[2]', coeffs[2])
        # print('final_term', add_with_constant(n, term, coeffs[2]))
        return add_with_constant(n, term, coeffs[2])
    else:
        return term
    # print('coeffs[2]', coeffs[2])
    # return add_with_constant(n, pA, coeffs[2])












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






    




    


    
