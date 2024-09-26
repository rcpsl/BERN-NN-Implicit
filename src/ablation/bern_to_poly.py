#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 23:48:21 2022

@author: waelfatnassi
"""



import numpy as np
import math
import torch
from poly_utils_cuda import *
from lower_upper_linear_ibf import *
import sys
sys.path.append('../')
from Bern_NN import SumModule


class multi_bern_to_pow_poly():
    
    def __init__(self, dims, intervals):
        
        self.dims = dims
        self.intervals = torch.tensor(intervals)


    # ========================================================
    #  given an index i and intervals =  [[d1min , d1max]....[dnmin , dnmax]]
    #  computes (d1max - d1min)...(dimax^2 - dimin^2)....(dnmax - dnmin)
    # ========================================================
    def prod_square_interval(self, index, intervals):

        res = 1
        for i in range(self.dims):
            if i == index:
                res = res * (intervals[i][1] ** 2 - intervals[i][0] ** 2)
            else:
                res = res * (intervals[i][1] - intervals[i][0])

        return res


    # ========================================================
    #  compute volume of input space defined by intervals
    #  vol = (d1max - d1min)....(dnmax - dnmin)
    # ========================================================
    def vol_input(self, intervals):

        vol = 1
        for i in range(self.dims):
            vol = vol * (intervals[i][1] - intervals[i][0])
        return vol




    # ========================================================
    #  given indices indice = [i1,...,in] compute product 
    #  of their inverse: 1/(i1 + 1)....(in + 1)
    # ========================================================
    def inverse_prod_indice(self, indice):

        res = 1
        for i in range(self.dims):
            res = res * (1 / (indice[i] + 1))
        
        # res = 1 / res
        # print(res)
        return res



    # ========================================================
    #  given indices indice = [i1,...,in] and 
    #  input's intervals, it computes product of difference of
    #  interval power that indice: (di_max)^i+1 - (di_min)^i+1
    # ========================================================
    def prod_diff_intervals(self, indice, intervals):
        res = 1
        for i in range(self.dims):
            res = res * (intervals[i][1] ** (indice[i] + 1) - intervals[i][0] ** (indice[i] + 1))
        return res



    # ========================================================
    #   obtain dims that are even and do reverse
    # ========================================================
    def dims_even(self):
        dims_even_list =  [i + 1 for i in range(self.dims) if (i + 1) % 2 == 0]
        dims_even_list.reverse()
        return dims_even_list


    # ========================================================
    #   obtain dims that are odd and do reverse
    # ========================================================
    def dims_odd(self):
        dims_odd_list =  [i + 1 for i in range(self.dims) if (i + 1) % 2 == 1]
        dims_odd_list.reverse()
        return dims_odd_list



    # ========================================================
    #   Compute the binomial of n choose k 
    # ========================================================
    def binom(self, n, k):
        return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

    def binom_list(self, n, k):
        a = torch.lgamma(n+1)
        b = torch.lgamma(n-k+1)
        c = torch.lgamma(k+1)
        return torch.exp(a-b-c)
    
    



    # ========================================================
    #   Compute Vx
    # ========================================================

    def Vx(self, order, interval):
        #print('interval', interval)
        #v = [1/((interval[1] - interval[0]) ** i) for i in range(self.order + 1)]
        v = 1/torch.pow(interval[1]-interval[0], torch.arange(order+1))
        v = torch.tensor(v, dtype = torch.float32)
        Vx = torch.diag(v)

        return Vx

    # ========================================================
    #   Compute Ux
    # ========================================================

    def Ux(self, order):

        Ux = []

        for i in range(order + 1):
            row = []
            for j in range(order + 1):

                if j <= i:
                    u = self.binom(order, j) * self.binom(order - j, i - j) * (-1) ** (i - j)
                else:
                    u = 0
                row.append(u)    

            Ux.append(row)    
        Ux = torch.tensor(Ux, dtype = torch.float32)
        Ux = torch.tril(Ux)

        return Ux        


    # ========================================================
    #   Compute Wx
    # ========================================================

    def Wx(self, order, interval):

        Wx = []

        I = torch.arange(order+1).reshape(order+1,1)
        J = torch.arange(order+1).reshape(1, order+1)
        diff = J - I
        binomial_coeffs = self.binom_list(J, I)
        Wx = binomial_coeffs * torch.pow(-interval[0], diff)
        
        Wx = torch.tensor(Wx, dtype = torch.float32)
        Wx = torch.triu(Wx)

        return Wx    



    # ========================================================
    #   Compute Mx = Wx Vx Ux
    # ========================================================
                  
            
    def Mx(self, order, interval):
        
        Mx = torch.matmul(self.Vx(order, interval), self.Ux(order))
        Mx = torch.matmul(self.Wx(order, interval), Mx)
            
        return Mx 


    # ========================================================
    #   convert bern_coeffs to pow_coeffs
    # ========================================================


    def convert(self, bern_coeffs, orders):

        # check if we do transpose of self.bern_coeffs by checking parity of self.dims
        do_transpose = self.dims % 2

        # compute Mx_i for every x_i and put it in a list Mx_list
        Mx_list = []

        for i in range(self.dims):
            Mx = self.Mx(orders[i], self.intervals[i])
            Mx_list.append(Mx)

        dims_even_rev_list = self.dims_even()
        dims_odd_rev_list = self.dims_odd()

        # compute right part which consists of product of reversed even dim i, M_i
        M_right = self.Mx(orders[dims_even_rev_list[0] - 1], self.intervals[dims_even_rev_list[0] - 1])
        if len(dims_even_rev_list) > 1:
            for i in range(1, len(dims_even_rev_list)):
                M_next = self.Mx(orders[dims_even_rev_list[i] - 1], self.intervals[dims_even_rev_list[i] - 1])
                M_right = torch.matmul(M_next, M_right) 

        # do transpose
        M_right = M_right.permute(*torch.arange(M_right.ndim-1, -1, -1))

        # compute left part which consists of product of reversed odd dim i, M_i
        M_left = self.Mx(orders[dims_odd_rev_list[0] - 1], self.intervals[dims_odd_rev_list[0] - 1])
        if len(dims_odd_rev_list) > 1:
            for i in range(1, len(dims_odd_rev_list)):
                M_next = self.Mx(orders[dims_odd_rev_list[i] - 1], self.intervals[dims_odd_rev_list[i] - 1])
                M_left = torch.matmul(M_next, M_left) 


        # compute pow_coeffs

        if do_transpose == 0:
            pow_coeffs = torch.matmul(bern_coeffs, M_right)
            pow_coeffs = torch.matmul(M_left, pow_coeffs)

        else:
            self.bern_coeffs = bern_coeffs.permute(*torch.arange(bern_coeffs.ndim-1, -1, -1))
            pow_coeffs = torch.matmul(self.bern_coeffs, M_right)
            pow_coeffs = torch.matmul(M_left, pow_coeffs)


        return pow_coeffs




    def convert_to_polyar_form(self):

        pow_coeffs = self.convert()
        shape = list(torch.tensor(pow_coeffs.shape))
        # print(shape)
        indices = torch.nonzero(torch.ones(shape))
        # print(indices)
        pow_coeffs_flatten = pow_coeffs.reshape(-1)



        polycs = []
        term = {}

        for i in range(len(pow_coeffs_flatten)):
            coeff = float(pow_coeffs_flatten[i])
            if coeff != 0 :
                varspows=[]
                for k in indices[i]:
                    varspows.append({'power': k})
                    
                term={'coeff':coeff,'vars': varspows}
                polycs.append(term)   
            
        return polycs


    # ========================================================
    #   padd inputs of degree to final_degree by adding 
    #   zeros to the remaining spaces
    # ========================================================

    def padding(self, inputs, dims, degree, final_degree):
        padding_amount = [lfd_i - ld_i for lfd_i, ld_i in zip(final_degree, degree)]
        padding_size = []
        for i in range(dims):
            padding_size += [0, int(padding_amount[dims-1-i])]
        padding_size = tuple(padding_size)
        return torch.nn.functional.pad(inputs, padding_size)
        


    # # ========================================================
    # #   compute the volume between the multivariate bernstein
    # #   functions
    # #   f(x1,...,xn):Rn--->R and g(x1,...,xn):Rn--->R
    # #   assume that f >= g
    # # ========================================================

    # def vol_between_f_g(self, g, f):
       
    #     g_degree = torch.tensor(g.shape) - 1
    #     f_degree = torch.tensor(f.shape) - 1
    #     # convert f and g to their power series representation
    #     f = self.convert(f, f_degree)
    #     g = self.convert(g, g_degree)

    #     ### get f - g
    #     # if f and g have the same shape just sum them
    #     if torch.all(f_degree == g_degree):
    #         diff_f_g = f - g

    #     else:
    #         # padd g
    #         g = self.padding(g, self.dims, g_degree, f_degree)
    #         diff_f_g = f - g

        
    #     # compute volume between f and g
    #     diff_f_g_shape = list(f_degree + 1)
    #     # print(shape)
    #     indices = torch.nonzero(torch.ones(diff_f_g_shape))
    #     # print(indices)
    #     diff_f_g_coeffs_flatten = diff_f_g.reshape(-1)

    #     vol = 0
        
    #     # print(diff_f_g_coeffs_flatten)
    #     for i in range(len(diff_f_g_coeffs_flatten)):
    #         coeff = diff_f_g_coeffs_flatten[i]
    #         res = coeff * self.inverse_prod_indice(indices[i])
    #         res = res * self.prod_diff_intervals(indices[i], self.intervals)
    #         vol = vol + res

    #     return vol


    # ========================================================
    #   compute the volume between the multivariate bernstein
    #   functions in ibf form
    #   f(x1,...,xn):Rn--->R and g(x1,...,xn):Rn--->R
    #   assume that f >= g
    # ========================================================

    def vol_between_f_g_ibf(self, g, f, g_degree, f_degree):
        #print('f', f)
        #print('g', g)
       
        #### tf = number of rows of f divided by number of variables
        #### tg = number of rows of g divided by number of variables
        tg = g.shape[0] // self.dims
        tf = f.shape[0] // self.dims
        # g_degree = torch.tensor(g.shape) - 1
        # f_degree = torch.tensor(f.shape) - 1

        #print('g_degree', g_degree)
        #print('f_degree', f_degree)
        ### get f - g
        # g = (-1.0) * g
        g = mult_with_constant(self.dims, g, -1)
        # sum_module = SumModule(self.dims, g_degree, f_degree)
        # diff_f_g = sum_module(g, f)
        diff_f_g = sum_2_polys(self.dims, g, tg, g_degree, f, tf, f_degree)

        ### convert diff_f_g to dense tensor
        diff_f_g_coeffs_flatten = ibf_to_dense(self.dims, diff_f_g)
        


        # print(diff_f_g)
        # print(diff_f_g.shape)
        
        # compute volume between f and g
        # diff_f_g_coeffs_flatten = diff_f_g.reshape(-1)

        vol = 0
        
        # print(diff_f_g_coeffs_flatten)
        for i in range(len(diff_f_g_coeffs_flatten)):
            coeff = diff_f_g_coeffs_flatten[i]
            res = coeff * self.vol_input(self.intervals) / (torch.prod((f_degree + 1)))
            vol = vol + res

        return vol


    def vol_between_f_g(self, g, f):
        g_degree = torch.tensor(g.shape) - 1
        f_degree = torch.tensor(f.shape) - 1

        ### get f - g
        g = (-1.0) * g
        sum_module = SumModule(self.dims, g_degree, f_degree)
        diff_f_g = sum_module(g, f)
        
        # compute volume between f and g
        diff_f_g_shape = list(f_degree + 1)
        # print(shape)
        indices = torch.nonzero(torch.ones(diff_f_g_shape))
        # print(indices)
        diff_f_g_coeffs_flatten = diff_f_g.reshape(-1)

        vol = 0
        
        for i in range(len(diff_f_g_coeffs_flatten)):
            coeff = diff_f_g_coeffs_flatten[i]
            res = coeff * self.vol_input(self.intervals) / ((f_degree[0] + 1) ** self.dims)
            vol = vol + res

        return vol

    # ========================================================
    #  compute relative volume of beta crown defined by corwn_intervals
    #  vol = (d1max - d1min)....(dnmax - dnmin)/ vol_input
    # ========================================================
    def crown_volume(self, corwn_intervals):

        vol = 1
        n_output = len(corwn_intervals)
        for i in range(n_output):
            vol = vol * (corwn_intervals[i][1] - corwn_intervals[i][0])
        
        return vol


    # ========================================================
    #   compute SIA relative volume
    # ========================================================
    def sia_volume(self, sia_linear_eqs):
        sia_lower_eqs = sia_linear_eqs[0]
        sia_upper_eqs = sia_linear_eqs[1]
        sia_diff = sia_upper_eqs - sia_lower_eqs
        n_outputs = len(sia_diff)
        
        vol = 1
        for i in range(n_outputs):
            sia_diff_coeffs = sia_diff[i][:-1]
            sia_diff_last_coeff = sia_diff[i][-1]
            vol_i = sia_diff_last_coeff * self.vol_input(self.intervals) + self.linear_integral(sia_diff_coeffs, self.intervals)
            vol = vol * vol_i

        vol = vol / self.vol_input(self.intervals)
        return vol



    # ========================================================
    #   compute Bern_NN_IBF volume
    # ========================================================
    def Bern_NN_IBF_volume(self, bern_ibf_under, bern_ibf_over, under_degrees, over_degrees):

        n_outputs = len(bern_ibf_over)

        vol = 1 
        for i in range(n_outputs):
            vol_i = self.vol_between_f_g_ibf(bern_ibf_under[i], bern_ibf_over[i], under_degrees[i], over_degrees[i])
            vol = vol * vol_i
            
        vol = vol / self.vol_input(self.intervals)
        return vol


    # ========================================================
    #   compute Bern_NN volume
    # ========================================================
    def Bern_NN_volume(self, bern_under, bern_over):

        n_outputs = len(bern_over)

        vol = 1 
        for i in range(n_outputs):
            vol_i = self.vol_between_f_g(bern_under[i], bern_over[i])
            vol = vol * vol_i
            
        vol = vol / self.vol_input(self.intervals)
        return vol


    # ========================================================
    #   compute integral of linear function a1x1+...+anxn
    #   where coeffs = [a1,...,an] and intervals of x1,..,xn  
    # ========================================================

    def linear_integral(self, coeffs, intervals):
        
        integ = 0
        for i in range(self.dims):
            integ_i = (coeffs[i] * 0.5) * (self.prod_square_interval(i, intervals))
            integ = integ + integ_i
        return integ








        












if __name__ == "__main__":


    dims = 2
    intervals = np.array([[-0.8, 0.9] , [-0.5, 0.6]])
    bern_coeffs_over = torch.tensor([[17.0328, 17.5662, 19.0961, 21.6772, 25.6703],
                                [17.9587, 19.4538, 21.9844, 25.7027, 31.0667],
                                [21.8251, 24.3489, 28.0520, 33.1840, 40.3005],
                                [28.9109, 32.8164, 38.1499, 45.2584, 54.7949],
                                [42.1417, 48.0679, 55.7757, 65.7095, 78.6195]], dtype = torch.float32)

    bern_coeffs_under = torch.tensor([[-19.5582,  -4.7350],
                                 [  7.6851,  22.5083]], dtype = torch.float32)




    from_ber_to_poly = multi_bern_to_pow_poly(dims, intervals)

    vol = from_ber_to_poly.vol_between_f_g(bern_coeffs_under, bern_coeffs_over)
    vol_input = from_ber_to_poly.vol_input(intervals)

    print(vol/vol_input)















        




            
        
            
    
