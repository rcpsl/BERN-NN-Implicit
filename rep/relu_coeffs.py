import numpy as np
import math
import torch



class relu_monom_coeffs():

    def __init__(self, intervals):
        self.intervals = intervals
        





    ######################################################################
    # get monomial coeffs for the upper polymial for the relu over interval
    #######################################################################
    
    def get_monom_coeffs_over(self, interval):

    
        if (interval[0] == interval[1]) and (interval[0] == 0.0):

            coeffs = torch.tensor([0., 0., 0.])

        else:

            U = torch.tensor([[1., 0., 0.], [-2., 2., 0.], [1., -2., 1.]])
            V = torch.diag(torch.tensor([1., 1 / (interval[1] - interval[0]), 1 / (interval[1] - interval[0]) ** 2]))
            W = torch.tensor([[1., (-1) * interval[0], interval[0] ** 2], [0, 1., (-1) * 2 * interval[0]], [0., 0., 1.]])

            
            bern_coeffs = torch.tensor([[torch.relu(interval[0])], [torch.relu(0.5 * (interval[0] + interval[1]))], [torch.relu(interval[1])]])
            coeffs = torch.matmul(U, bern_coeffs)
            coeffs = torch.matmul(V, coeffs)
            coeffs = torch.matmul(W, coeffs)

            coeffs = coeffs.reshape(-1)
            ### switch the places of coeffs[0] and coeffs[2] to get the coeffs in the right order
            coeffs = coeffs.flip(dims=[0])



            # coeffs = torch.tensor([0., interval[1] / (interval[1] - interval[0]), - (interval[1] * interval[0]) / (interval[1] - interval[0])])
            # coeffs = torch.tensor([1., 0., 0.])
            # coeffs = torch.tensor([0., interval[1] / (interval[1] - interval[0]), (- interval[1] * interval[0])  / (interval[1] - interval[0])])

        return coeffs




    ######################################################################
    # get monomial coeffs for the lower polymial for the relu over interval
    #######################################################################
    
    def get_monom_coeffs_under(self, interval):

    
        if (interval[0] == interval[1]) and (interval[0] == 0):

            coeffs = torch.tensor([0., 0., 0.])
            return coeffs

        else:
            
  
            # f1 = 0.
            # f2 = (2 * interval[1] ** 2 - interval[1] * interval[0] - interval[0] ** 2) / 6.0
            # f3 = (interval[1] ** 2 - interval[0] ** 2) / 2.0

            # if max(f1, f2, f3) == f1:
            #     return torch.tensor([0., 0., 0.])

            # elif max(f1, f2, f3) == f2:
            #     return torch.tensor([1 / (interval[1] - interval[0]), -interval[0] / (interval[1] - interval[0]), 0.])

            # else:
            #     return torch.tensor([0., 1., 0.])

            return torch.tensor([0., 0., 0.])





    