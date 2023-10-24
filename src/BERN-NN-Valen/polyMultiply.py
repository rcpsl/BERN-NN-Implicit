'''
Author: Haitham Khedr
Sample code for multiplying two multi-variate polynomials
'''

from collections import defaultdict
import torch
from time import time
from numba import cuda
import numpy as np
import os
EPS = 1E-10
def read_poly(fname, n_terms):
    exp = []
    coeffs = []
    with open(fname,'r') as f:
        lines  = f.readlines()[:n_terms]
        for line in lines:
            nums = line.split()
            exp.append(list(map(int,nums[:-1])))
            coeffs.append(float(nums[-1]))
    return torch.tensor(exp,dtype = torch.uint8), torch.tensor(coeffs,dtype = torch.float32)

def multi_dim_binom(degree):
    if(isinstance(degree, list)):
        degree = torch.hstack(degree)
    dims = len(degree)
    sizes = torch.Size((degree + 1).long())
    res = torch.ones(sizes, dtype=torch.float32)   
    indices = torch.nonzero(res)
    N = degree
    R = indices
    a = torch.lgamma(N+1)
    b = torch.lgamma(N-R+1)
    c = torch.lgamma(R+1)
    res = torch.exp(torch.sum(a-b-c, 1)).reshape(sizes)
    
    """
    for ind in indices:
        res[tuple(ind)] = multi_dim_nCr(degree, ind)
    """
    return res    

def poly_multiplication(polyA, polyB):

    #device = polyA.device
    final_shape = torch.Size(torch.tensor(polyA.shape) + torch.tensor(polyB.shape) - 1)
    pA, pA_coeff = torch.nonzero(torch.ones(polyA.shape)), torch.flatten(polyA)
    pB, pB_coeff = torch.nonzero(torch.ones(polyB.shape)), torch.flatten(polyB)
    n_vars = pA.shape[-1]
    hash_factor = torch.arange(n_vars-1,-1,-1, dtype = torch.long)
    #pA = pA.to(device)
    #pB = pB.to(device)
    #pA_coeff = pA_coeff.to(device)
    #pB_coeff = pB_coeff.to(device)
    #hash_factor = hash_factor.to(device)
    pC = pA + pB.unsqueeze(1) # -> NA_terms X NB_terms X N_vars
    pC = pC.reshape(-1, n_vars) #->  (NA_terms * NB_terms) X N_vars
    pC_coeff = (pA_coeff * pB_coeff.unsqueeze(1)).flatten() #Compute result coefficients
    #max_order = torch.tensor(129,dtype=torch.uint8,device = device) #Max allowable order 
    max_order = torch.tensor(129,dtype=torch.uint8) #Max allowable order 
    hash_factor = torch.pow(max_order,hash_factor)
    hash_vals = torch.sum(pC * hash_factor,dim=1) #hash value for each term 

    unique_exp, mapping = hash_vals.unique(return_inverse =True)
    #perm = torch.arange(mapping.size(-1), dtype=mapping.dtype, device=mapping.device)
    perm = torch.arange(mapping.size(-1), dtype=mapping.dtype)
    inverse, perm = mapping.flip([-1]), perm.flip([-1])
    unique_exp = inverse.new_empty(unique_exp.size(-1)).scatter_(-1, inverse, perm)
    result_exp = pC[unique_exp] 
    #result_coeff = torch.zeros(unique_exp.shape[0], device=device)
    result_coeff = torch.zeros(unique_exp.shape[0])
    result_coeff.scatter_add_(-1,mapping,pC_coeff)
    final_result = result_coeff.reshape(final_shape)
    return final_result


def nCr_order(order):
    # Computes orderCr for r in [0,order]
    # nCr = nCr-1 * (n-r+1) /r
    result = [1]
    for i in range(1,order+1):
        nci = result[-1] * (order - i +1) / i
        result.append(nci)
    return torch.tensor(result, dtype = torch.float32)


def get_bern_scaling_coeffs(poly):
    order = torch.tensor(poly.shape) - 1
    combinations = []
    for order_dim in order:
        combinations.append(nCr_order(order_dim))
    scaling_coeffs = combinations[0]
    for comb in combinations[1:]:
        scaling_coeffs = scaling_coeffs.unsqueeze(-1) * comb    
    
    return scaling_coeffs
def scaled_bern(poly):
    scaling_coeffs = get_bern_scaling_coeffs(poly).to(poly.device)
    scaled_poly = poly * scaling_coeffs
    return scaled_poly

def unscaled_bern(scaled_poly):
    scaling_coeffs = get_bern_scaling_coeffs(scaled_poly).to(scaled_poly.device)
    poly = scaled_poly / scaling_coeffs # scaling_coeffs will never have zero terms
    return poly

def bern_multiply(polyA, polyB):
    """
    scaled_polyA  = scaled_bern(polyA)
    scaled_polyB  = scaled_bern(polyB)
    scaled_result = poly_multiplication(scaled_polyA, scaled_polyB)
    result = unscaled_bern(scaled_result)
    return result
    """
    scaling_A = multi_dim_binom(torch.tensor(polyA.shape)-1)
    scaling_B = multi_dim_binom(torch.tensor(polyB.shape)-1)
    scaled_A = scaling_A * polyA
    scaled_B = scaling_B * polyB
    result = poly_multiplication(scaled_A, scaled_B)
    unscaling = multi_dim_binom(torch.tensor(result.shape)-1)
    return result / unscaling

def poly_multiplication2(pA, pB):
    final_shape = pA.shape
    pA, pA_coeff = tf.where(tf.ones(pA.shape)), tf.reshape(pA, [-1])
    pB, pB_coeff = tf.where(tf.ones(pB.shape)), tf.reshape(pB, [-1])
    n_vars = pA.shape[-1]
    hash_factor = tf.range(n_vars-1,-1,-1, dtype =tf.int64)
    pC = pA + tf.expand_dims(pB, 1) # -> NA_terms X NB_terms X N_vars
    pC = tf.reshape(pC, (-1, n_vars)) #->  (NA_terms * NB_terms) X N_vars
    pC_coeff = tf.reshape((pA_coeff * tf.expand_dims(pB_coeff, 1)), [-1]) #Compute result coefficients
    max_order = tf.constant(200,dtype=tf.int64) #Max allowable order 
    hash_factor = tf.math.pow(max_order,hash_factor)
    hash_vals = tf.math.reduce_sum(pC * hash_factor, 1) #hash value for each term 

    unique_exp, mapping = tf.unique(hash_vals)
    print(unique_exp, mapping)
    perm = tf.range(mapping.shape[0], dtype=mapping.dtype)
    unique_exp = tf.zeros(unique_exp.shape[0], dtype=tf.int32)
    unique_exp = tf.tensor_scatter_nd_update(unique_exp, tf.expand_dims(mapping, 1), perm)
    print(unique_exp)
    result_exp = tf.gather(pC, unique_exp)
    print(result_exp)
    result_coeff = tf.zeros(unique_exp.shape[0])
    #print(result_coeff.dtype, mapping.dtype, pC_coeff.dtype)
    result_coeff = tf.tensor_scatter_nd_add(result_coeff, tf.expand_dims(mapping, 1), pC_coeff)
    print(result_coeff)
    result_coeff = tf.zeros(unique_exp.shape[0])
    final_result = tf.zeros(final_shape, dtype=tf.float32)
    final_result = tf.tensor_scatter_nd_update(final_result, result_exp, result_coeff)
    """
    for i in range(result_exp.shape[0]):
        if result_coeff[i] != 0:
            final_result[tuple(result_exp[i])] = result_coeff[i]
    """
    return final_result




if __name__ == "__main__":
    print("\n")
    device = 'cuda'

    DIR = 'Tests/pol_files/3var10or/'
    DIR = '/home/simulator/composition/MultivariatePolynomialMultiplicationGPU/Tests/pol_files/3var10or'
    pA, pA_coeff = read_poly(os.path.join(DIR,'a'), n_terms = 4096)
    pB, pB_coeff = read_poly(os.path.join(DIR,'a'), n_terms = 4096)
    print("HJIHIII", pA.shape, pA_coeff.shape)
    n_vars = pA.shape[-1]
    #used later for hashing a vector of exponent to a unique value.
    hash_factor = torch.arange(n_vars-1,-1,-1, dtype = torch.long)
    
    #warmup (first kernel call is slow)
    if(device == 'cuda'):
        s = time()
        temp = torch.zeros(100,100)
        temp = temp.to(device)
        temp = temp.to('cpu')
        e = time()
        print(f"Warmup time:{1000*(e-s):.4f} ms")
        
    
    transfer_time = 0
    if(device == 'cuda'):
        s = time()
        pA = pA.to(device)
        pB = pB.to(device)
        pA_coeff = pA_coeff.to(device)
        pB_coeff = pB_coeff.to(device)
        hash_factor = hash_factor.to(device)
        e = time()
        transfer_time = e-s
        print(f"GPU MemCpy time:{1000*transfer_time:.4f} ms")


    #Computation (multiplication + hash computation) pC = pA * pB
    s = time()



    pC = pA + pB.unsqueeze(1) # -> NA_terms X NB_terms X N_vars
    pC = pC.reshape(-1, n_vars) #->  (NA_terms * NB_terms) X N_vars
    pC_coeff = (pA_coeff * pB_coeff.unsqueeze(1)).flatten() #Compute result coefficients
    max_order = torch.tensor(65,dtype=torch.uint8,device = device) #Max allowable order 
    hash_factor = torch.pow(max_order,hash_factor)
    hash_vals = torch.sum(pC * hash_factor,dim=1) #hash value for each term 
    
    e = time()
    compute_time = e-s
    print(f"Computation time:{1000*compute_time:.4f} ms")

    #Reduction (add terms with same exponents)
    s = time()
    #TODO: optimize this, `.unique()` very expensive
    #Goal is to Get the unique indices not elements (not supported by pytorch :( )
    unique_exp, mapping = hash_vals.unique(return_inverse =True)
    perm = torch.arange(mapping.size(-1), dtype=mapping.dtype, device=mapping.device)
    inverse, perm = mapping.flip([-1]), perm.flip([-1])
    #unique_exp is the index of terms with unique exponent values
    unique_exp = inverse.new_empty(unique_exp.size(-1)).scatter_(-1, inverse, perm)
    #########################
    result_exp = pC[unique_exp] 
    result_coeff = torch.zeros(unique_exp.shape[0],device = device)
    result_coeff.scatter_add_(-1,mapping,pC_coeff)
    e = time()
    
    reduce_time = e - s
    print(f"Reduction time:{1000*reduce_time:.4f} ms")

    #Copy back to CPU
    if(device == 'cuda'):
        s = time()
        result_exp  = result_exp.to('cpu')
        result_coeff = result_coeff.to('cpu')

    e = time()
    copy_time = e-s 
    print(f"Device to host copy time:{1000*copy_time:.4f} ms")
    print(f"Total time:{1000*(transfer_time +  compute_time + reduce_time + copy_time):.4f} ms")
    WRITE_TO_FILE = True
    if(WRITE_TO_FILE):
        fname = os.path.join(DIR,'result')
        with open(fname,'w') as f:
            for exp, coeff in zip(result_exp, result_coeff):
                for val in exp:
                    f.write(f"{val} ")
                f.write(f"{coeff}\n")
            
            f.close()
