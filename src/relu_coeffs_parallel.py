import numpy as np
import math
import torch



class relu_monom_coeffs():

    def __init__(self, intervals, device):
        self.intervals = intervals
        self.device = device
        





    ######################################################################
    # get monomial coeffs for the upper polymial for the relu over interval
    #######################################################################
    
    def get_monom_coeffs_over(self, interval):
        # Use the device attribute of the class instance
        device = self.device

        if (interval[0] == interval[1]) and (interval[0] == 0.0):
            coeffs = torch.tensor([0., 0., 0.], device=device)
        else:
            U = torch.tensor([[1., 0., 0.], [-2., 2., 0.], [1., -2., 1.]], device=device)
            V = torch.diag(torch.tensor([1., 1 / (interval[1] - interval[0]), 1 / (interval[1] - interval[0]) ** 2], device=device))
            W = torch.tensor([[1., (-1) * interval[0], interval[0] ** 2], [0, 1., (-1) * 2 * interval[0]], [0., 0., 1.]], device=device)

            # Ensure the interval is also on the correct device
            interval = interval.to(device)
            
            bern_coeffs = torch.tensor([[torch.relu(interval[0])], [torch.relu(0.5 * (interval[0] + interval[1]))], [torch.relu(interval[1])]], device=device)
            coeffs = torch.matmul(U, bern_coeffs)
            coeffs = torch.matmul(V, coeffs)
            coeffs = torch.matmul(W, coeffs)

            coeffs = coeffs.reshape(-1)
            # Switch the places of coeffs[0] and coeffs[2] to get the coeffs in the right order
            coeffs = coeffs.flip(dims=[0])

        return coeffs





    ######################################################################
    # get monomial coeffs for the lower polymial for the relu over interval
    #######################################################################
    
    def get_monom_coeffs_under(self, interval):
        # Use the device attribute of the class instance
        device = self.device

        if (interval[0] == interval[1]) and (interval[0] == 0):
            coeffs = torch.tensor([0., 0., 0.], device=device)
            return coeffs
        else:
            interval = interval.to(device)  # Ensure interval is on the correct device

            f1 = 0.
            f2 = (2 * interval[1] ** 2 - interval[1] * interval[0] - interval[0] ** 2) / 6.0
            f3 = (interval[1] ** 2 - interval[0] ** 2) / 2.0

            if max(f1, f2, f3) == f1:
                return torch.tensor([0., 0., 0.], device=device)

            elif max(f1, f2, f3) == f2:
                return torch.tensor([1 / (interval[1] - interval[0]), -interval[0] / (interval[1] - interval[0]), 0.], device=device)

            else:
                return torch.tensor([0., 1., 0.], device=device)







    def get_monom_coeffs_layer(self):


        coeffs_layer_over = []
        coeffs_layer_under = []
        #print(self.intervals)
        for interval in self.intervals:   
            #print(interval)
            if interval[1] <= 0:

                coeffs_zeros = torch.tensor([0.0, 0.0, 0.0])
                coeffs_over = coeffs_zeros
                coeffs_under = coeffs_zeros
                coeffs_layer_over.append(coeffs_over)
                coeffs_layer_under.append(coeffs_under)

            elif interval[0] >= 0:

                #coeffs_over = self.get_monom_coeffs_over(interval)
                coeffs_over = torch.tensor([0.0, 1.0, 0.0])
                # round coeffs_over to a 10 decimals after ,: 0.000000000d
                #coeffs_over = self.round_list(coeffs_over, 10)

                coeffs_under = coeffs_over
                coeffs_layer_over.append(coeffs_over)
                coeffs_layer_under.append(coeffs_under)

            else:


                coeffs_over = self.get_monom_coeffs_over(interval)
                coeffs_under = self.get_monom_coeffs_under(interval)
                
                # if abs(interval[0]) == interval[1]:
                #     coeffs_under = self.get_monom_coeffs_under(interval)

                # if (interval[1] > abs(interval[0])):
                #     #coeffs_under = self.get_monom_coeffs_under(interval)
                #     coeffs_under = [0.0, 1.0] + [0.0] * ((self.order + 1) - 2)
                #     coeffs_under = torch.tensor(coeffs_under)
                # if (interval[1] < abs(interval[0])):    
                #     coeffs_under = [0.0] * (self.order + 1)
                #     coeffs_under = torch.tensor(coeffs_under)

                # print('intervalintervalintervalintervalintervalintervalintervalintervalintervalintervalintervalinterval', interval)
                coeffs_layer_over.append(coeffs_over)
                coeffs_layer_under.append(coeffs_under)



                




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
            


