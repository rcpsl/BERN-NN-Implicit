#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 23:48:21 2022

@author: waelfatnassi
"""



import numpy as np
import math
import torch


class relu_monom_coeffs():
    
    def __init__(self, order, intervals, bounds_lower_bern):
         
        self.order = order
        self.intervals = intervals
        self.bounds_lower_bern = bounds_lower_bern


    # ========================================================
    #   round all the items in a list L to a certain number
    #   decimals m
    # ======================================================== 
    def round_list(self, L, m):
        
        res = []
        for item in L:

            item_rounded = round(item, m)
            res.append(item_rounded)

        return res    


    
    
    # ========================================================
    #   remove all the zero columns from the end of a matrix
    # ========================================================
    def pop_zeros(self, matrix):
        while np.all(matrix[:, -1] == 0):
            matrix = np.delete(matrix, -1, axis=1)
        return matrix  

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
    #   Compute relu of x
    # ========================================================
    def relu(self, x):
        
        return (abs(x) + x) / 2


    # ========================================================
    #   Compute Vx
    # ========================================================

    def Vx(self, interval):
        #print('interval', interval)
        #v = [1/((interval[1] - interval[0]) ** i) for i in range(self.order + 1)]
        v = 1/torch.pow(interval[1]-interval[0], torch.arange(self.order+1))

        Vx = torch.diag(v)

        return Vx

    # ========================================================
    #   Compute Ux
    # ========================================================

    def Ux(self):

        Ux = []

        for i in range(self.order + 1):
            row = []
            for j in range(self.order + 1):

                if j <= i:
                    u = self.binom(self.order, j) * self.binom(self.order - j, i - j) * (-1) ** (i - j)
                else:
                    u = 0
                row.append(u)    

            Ux.append(row)    

        Ux = torch.tril(torch.tensor(Ux, dtype=torch.float32))

        return Ux        


    # ========================================================
    #   Compute Wx
    # ========================================================

    def Wx(self, interval):

        Wx = []

        I = torch.arange(self.order+1).reshape(self.order+1,1)
        J = torch.arange(self.order+1).reshape(1, self.order+1)
        diff = J - I
        binomial_coeffs = self.binom_list(J, I)
        Wx = binomial_coeffs * torch.pow(-interval[0], diff)


        """
        for i in range(self.order + 1):
            row = []

            for j in range(self.order + 1):

                if j >= i:
                    w = self.binom(j, i)  * (-1 * interval[0]) ** (j - i)
                else:
                    w = 0    
                row.append(w)
            Wx.append(row)    
        """

        Wx = torch.triu(Wx)

        return Wx    







        
    # ========================================================
    #   Compute error 
    # ========================================================

    def error(self, interval, bern_coeffs):
        
        all_bern_base_0 = []

        for k in range(self.order + 1):

            val = self.binom(self.order, k) * ( ((-interval[0]) ** k) * (interval[1]) ** (self.order - k)    ) / ( (interval[1] - interval[0]) ** self.order    )
            all_bern_base_0.append(val) 


        error = sum(    [bern_coeffs[i] * all_bern_base_0[i]  for i in range(self.order + 1)]       )   

        return error 






    
    
    
    
    def relu_bern_coefficients(self, interval, order):
        
        #coeffs = []
        coeffs = (interval[1] - interval[0]) * torch.arange(order+1) / order
        coeffs = torch.nn.functional.relu(coeffs + interval[0])
        """
        for k in range(order + 1):
            #print(interval)
            #print(interval[0])
            #print(interval[1])
            c = self.relu((interval[1] - interval[0]) * (k / order) + interval[0])
            coeffs.append(c)
        """
            
            
        return coeffs
        return torch.tensor(coeffs)
            
            
            
    
    
          
            
    def get_monom_coeffs_over(self, interval):
        
        if (interval[0] == interval[1]) and (interval[0] == 0):

            coeffs = [0 for _ in range(self.order + 1)]

        else:

            bern_coeffs = self.relu_bern_coefficients(interval, self.order)
            bern_coeffs = bern_coeffs.reshape(-1, 1)
            #print('bern_coeffs', bern_coeffs)
            #print('self.Ux', self.Ux())
            #print('self.Vx', self.Vx(interval))
            #print('self.Wx', self.Wx(interval))
            coeffs = torch.matmul(self.Ux(), bern_coeffs)
            coeffs = torch.matmul(self.Vx(interval), coeffs)
            coeffs = torch.matmul(self.Wx(interval), coeffs)
            
            coeffs = coeffs.reshape(-1)
            #coeffs = list(coeffs)   

        return coeffs  


    def get_monom_coeffs_under(self, interval):
        
        if (interval[0] == interval[1]) and (interval[0] == 0):

            coeffs = [0 for _ in range(self.order + 1)]

        else:

            bern_coeffs = self.relu_bern_coefficients(interval, self.order)
            error = self.error(interval, bern_coeffs)
            bern_coeffs_under = [bern_coeffs[i] - error for i in range(self.order + 1)]
            bern_coeffs_under = torch.tensor(bern_coeffs_under).reshape(-1, 1)
            #print('bern_coeffs', bern_coeffs)
            #print('self.Ux', self.Ux())
            #print('self.Vx', self.Vx(interval))
            #print('self.Wx', self.Wx(interval))
            coeffs = torch.matmul(self.Ux(), bern_coeffs_under)
            coeffs = torch.matmul(self.Vx(interval), coeffs)
            coeffs = torch.matmul(self.Wx(interval), coeffs)
            
            coeffs = coeffs.reshape(-1)
            #coeffs = list(coeffs)   

        return coeffs      
    
    
    
    def get_monom_coeffs_layer(self):
        
       
        coeffs_layer_over = []
        coeffs_layer_under = []
        #print(self.intervals)
        i = 0
        for interval in self.intervals:   
            #print(interval)
            if interval[1] <= 0:

                coeffs_zeros = torch.tensor([0.0] * (self.order + 1))
                coeffs_over = coeffs_zeros
                coeffs_under = coeffs_zeros
                coeffs_layer_over.append(coeffs_over)
                coeffs_layer_under.append(coeffs_under)

            elif interval[0] >= 0:

                #coeffs_over = self.get_monom_coeffs_over(interval)
                coeffs_over = torch.tensor([0.0, 1.0] + [0.0] * ((self.order + 1) - 2))
                # round coeffs_over to a 10 decimals after ,: 0.000000000d
                #coeffs_over = self.round_list(coeffs_over, 10)

                coeffs_under = coeffs_over
                coeffs_layer_over.append(coeffs_over)
                coeffs_layer_under.append(coeffs_under)

            else:
       
                # print('intervalintervalintervalintervalintervalintervalintervalintervalintervalintervalintervalinterval', interval)
                coeffs_over = self.get_monom_coeffs_over(interval)
                # print('coeffs_overcoeffs_overcoeffs_overcoeffs_overcoeffs_overcoeffs_overcoeffs_overcoeffs_overcoeffs_over', coeffs_over)
                # coeffs_under = self.get_monom_coeffs_under(interval)
                # print('coeffs_undercoeffs_undercoeffs_undercoeffs_undercoeffs_undercoeffs_undercoeffs_undercoeffs_under', coeffs_under)
                # coeffs_under = [0.0] * (self.order + 1)
                # coeffs_under = torch.tensor(coeffs_under)
                
                # if abs(interval[0]) == interval[1]:
                #     coeffs_under = self.get_monom_coeffs_under(interval)

                if (interval[1] > abs(interval[0])):
                    #coeffs_under = self.get_monom_coeffs_under(interval)
                    if self.bounds_lower_bern[i][1] > 0:
                        slope = self.bounds_lower_bern[i][1] / (self.bounds_lower_bern[i][1] - self.bounds_lower_bern[i][0])
                        # print('slopeslopeslopeslopeslopeslopeslopeslopeslope', slope)
                        # slope = interval[1] / (interval[1] - interval[0])
                        coeffs_under = [0.0, slope] + [0.0] * ((self.order + 1) - 2)
                        coeffs_under = torch.tensor(coeffs_under)

                    else:
                        coeffs_under = [0.0] * (self.order + 1)
                        coeffs_under = torch.tensor(coeffs_under)


                if (interval[1] < abs(interval[0])):    
                    coeffs_under = [0.0] * (self.order + 1)
                    coeffs_under = torch.tensor(coeffs_under)

                # print('intervalintervalintervalintervalintervalintervalintervalintervalintervalintervalintervalinterval', interval)
                coeffs_layer_over.append(coeffs_over)
                coeffs_layer_under.append(coeffs_under)

            i = i + 1

                



        
        coeffs_layer_over = torch.stack(coeffs_layer_over)
        coeffs_layer_under = torch.stack(coeffs_layer_under)

        

        #if np.any(coeffs_layer_over):

        #    coeffs_layer_over = self.pop_zeros(coeffs_layer_over) 

        #if np.any(coeffs_layer_under):

        #    coeffs_layer_under = self.pop_zeros(coeffs_layer_under)     

        #order = np.shape(coeffs_layer)[1] 
        
        #print('coeffs_layer', coeffs_layer)
        """
        coeffs_layer_over = list(coeffs_layer_over)
        coeffs_layer_under = list(coeffs_layer_under)
        """



     


            
        return coeffs_layer_over, coeffs_layer_under
            
        
            
    
