import torch


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
def add_with_constant(n, pA, c):
    ### replace the non-zero elements for the first n rows of pA with 1s
    c_tensor = torch.where(pA[0 : n, :] != 0, 1, 0)
    ### multiply the first row of c_tensor with c
    c_tensor[0, :] = c_tensor[0, :] * c
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
def mult_with_constant(n, pA, tA, degree_A, c):

    result = pA.clone()  # Create a copy to avoid modifying the original tensor
    
    # Get the number of rows in result
    num_rows = result.size(0)
    
    # Create a mask to select every nth row
    mask = torch.arange(num_rows) % n == 0
    
    # Multiply selected rows with c
    result[mask] *= c

    return result



def degree_elevation(pA, n, tA, degree, new_degree):

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
    result = [torch.nn.functional.conv1d(res[i, :].reshape(1, -1).reshape(1, 1, -1),      torch.flip(C_diff[i, :].reshape(1, -1), dims=(1,)).reshape(1, 1, -1), padding = torch.max(l_diff).item()   ).flatten()             for i in range(res.size(0) )]
    # print(result)
    result = torch.stack(result)

    
    ### perform row-wise division between res_conv and C_new_degree for non-zero elements of C_new_degree
    result =  torch.where(C_new_degree != 0, torch.div(result, C_new_degree), 0)
    
    ### pad result tensor with 0 columns of number max(new_degree) - max(degree)

    pA_elevated = torch.nn.functional.pad(pA, (0, torch.max(l_diff)))

    ### replace the rows of pA_elevated with the rows of result given by the row indices idx_list

    # print(pA_elevated)
    # print(idx_list)
    # print(result)
    if len(I_non_zero) == n:
        pA_elevated = result
    else:
        pA_elevated[idx_list, :] = result

    return pA_elevated




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
     
    ### concatenate pA and pB 
    if torch.all(degree_A == degree_B):
        return torch.cat((pA, pB), 0)


    ### degree elevate pA concatenate it with pB
    elif torch.all(torch.ge(degree_B, degree_A) ):
        pA_elevated = degree_elevation(pA, n, tA, degree_A, degree_B)
        return torch.cat((pA_elevated, pB), 0)

    ### degree elevate pB concatenate it with pA
    elif torch.all(torch.ge(degree_A, degree_B) ):
        pB_elevated = degree_elevation(pB, n, tB, degree_B, degree_A)
        return torch.cat((pA, pB_elevated), 0)


    ### degree elevate pA and pB concatenate it with pB
    else:
        degree_max = torch.max(degree_A, degree_B)
        pA_elevated = degree_elevation(pA, n, tA, degree_A, degree_max)
        pB_elevated = degree_elevation(pB, n, tB, degree_B, degree_max)
        return torch.cat((pA_elevated, pB_elevated), 0)


###################################################
### performs multiplicaion between 2 2D terms  
###  tensors TA and TB
###################################################
def mult_2_terms(n, TA, C_A, TB, C_B, C_AmulB):

    ### perform multpilcation point-wise between TA and CA; TB and CB
    T_C_A = torch.mul(TA, C_A)
    T_C_B = torch.mul(TB, C_B)

    ### perform row-wise convolution between T_C_A and T_C_B
    res = [torch.nn.functional.conv1d(T_C_A[i, :].reshape(1, -1).reshape(1, 1, -1),      torch.flip(T_C_B[i, :].reshape(1, -1), dims=(1,)).reshape(1, 1, -1), padding = T_C_B.size(1) - 1   ).flatten()             for i in range(n)]
    ### stack the rows
    res = torch.stack(res) 
    
    ### perform row-wise division between res and C_AmulB for non-zero elements of C_AmulB
    res =  torch.where(C_AmulB != 0, torch.div(res, C_AmulB), 0)

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

    ### perform multiplication between each term of pA to all the terms of pB
    res = [mult_2_terms(n, pA[i * n : (i + 1) * n, :], C_A, pB[j * n : (j + 1) * n, :], C_B, C_AmulB) for j in range(tB)   for i in range(tA)]
    
    # print(res)
    ### stack the rows and reshape the final result to (number_of rows, torch.max(degree_mul) + 1)
    res = torch.stack(res).reshape(-1, torch.max(degree_mul) + 1)

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

    ### compute pA ** 2
    # print(pA)
    # print(degree_A)
    # print(tA)
    # print(pA.size(0) // n)
    res = mult_2_polys(n, pA, tA, degree_A,  pA, tA, degree_A)
    ### compute a * pA ** 2
    term1 = mult_with_constant(n, res, tA, degree_A, coeffs[0])
    ### compute b * pA 
    term2 = mult_with_constant(n, pA, tA, degree_A, coeffs[1])
    ### sum term1 + term2: a * pA ** 2 + b * pA 
    t_term1 = term1.size(0) // n
    degree_term1 = 2 * degree_A
    term = sum_2_polys(n, term1, t_term1, degree_term1, term2, tA, degree_A)
    ### sum term + c; a * pA ** 2 + b * pA + c
    res = add_with_constant(n, term, coeffs[2])
    return res












if __name__ == "__main__":


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
    n = 2
    pA = torch.tensor([[1., 2., 3], [4., 8., 3]])
    tA = 1
    degree_A = torch.tensor([2, 2])
    coeffs = torch.tensor([1, 2, 3])
    res = quad_of_poly(n, pA, tA, degree_A, coeffs)
    print(res)

    




    


    