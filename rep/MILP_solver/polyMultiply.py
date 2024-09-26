'''
Author: Haitham Khedr
Sample code for multiplying two multi-variate polynomials
'''

from collections import defaultdict
import torch
import tensorflow as tf
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

def poly_multiplication(polyA, polyB):

    device = polyA.device
    final_shape = torch.Size(torch.tensor(polyA.shape) + torch.tensor(polyB.shape) - 1)
    pA, pA_coeff = torch.nonzero(torch.ones(polyA.shape)), torch.flatten(polyA)
    pB, pB_coeff = torch.nonzero(torch.ones(polyB.shape)), torch.flatten(polyB)
    n_vars = pA.shape[-1]
    hash_factor = torch.arange(n_vars-1,-1,-1, dtype = torch.long)
    pA = pA.to(device)
    pB = pB.to(device)
    pA_coeff = pA_coeff.to(device)
    pB_coeff = pB_coeff.to(device)
    hash_factor = hash_factor.to(device)
    pC = pA + pB.unsqueeze(1) # -> NA_terms X NB_terms X N_vars
    pC = pC.reshape(-1, n_vars) #->  (NA_terms * NB_terms) X N_vars
    pC_coeff = (pA_coeff * pB_coeff.unsqueeze(1)).flatten() #Compute result coefficients
    max_order = torch.tensor(129,dtype=torch.uint8,device = device) #Max allowable order 
    hash_factor = torch.pow(max_order,hash_factor)
    hash_vals = torch.sum(pC * hash_factor,dim=1) #hash value for each term 

    unique_exp, mapping = hash_vals.unique(return_inverse =True)
    perm = torch.arange(mapping.size(-1), dtype=mapping.dtype, device=mapping.device)
    inverse, perm = mapping.flip([-1]), perm.flip([-1])
    unique_exp = inverse.new_empty(unique_exp.size(-1)).scatter_(-1, inverse, perm)
    result_exp = pC[unique_exp] 
    result_coeff = torch.zeros(unique_exp.shape[0], device=device)
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
    # print(order)
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

    scaled_polyA  = scaled_bern(polyA)
    scaled_polyB  = scaled_bern(polyB)
    scaled_result = poly_multiplication(scaled_polyA, scaled_polyB)
    result = unscaled_bern(scaled_result)
    return result

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
