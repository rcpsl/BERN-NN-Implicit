import torch
import numpy as np

from torch_modules import PowerModule, degree_elevation
from relu_lists import relu, find_range

binom_cache = dict()


class ReLUModule(torch.nn.Module):
    """
    Module that takes a polynomial in the form of a list of coefficients and then
    computes the composition of it and the incoming polynomial
    """

    def __init__( self, dims, coefficients, degree, gpus):
        """
        coefficients: list-like, list of coefficients for each term in composing polynomial, size will be degree + 1
        degree: int; degree of the input polynomial + 1
        terms: int; number of xi terms in input polynomial
        """
        super().__init__()
        
        self.dims = dims
        self.coefficients = coefficients
        self.degree = degree
        self.gpus = gpus
        self.modules = []

        for power, coef in enumerate( self.coefficients ):
            if coef != 0:
                self.modules.append( (coef, PowerModule( self.dims, self.degree, power, self.gpus) ) )
        self.powerModules = self.modules

    def forward( self, inputs ):
        final_degree = self.degree * (len(self.coefficients) - 1)
        total = 0
        if torch.any(self.coefficients):
            shp = self.dims * [1]
            for coef, module in self.powerModules:
                if coef != 0:
                    result = coef * module(inputs)
                    degree_elevated = degree_elevation(result, final_degree)
                    total += degree_elevation(result, final_degree)
        else:
            return torch.zeros(torch.Size(final_degree+1))
        return total

torch.set_default_tensor_type('torch.cuda.FloatTensor')
module = ReLUModule(torch.tensor(2).cuda(), torch.tensor([4, 1, 2]).cuda(), torch.tensor([2, 2]).cuda(), 1)
#a = torch.tensor([[4., 7., 2.], [10., 16., 5.], [6., 9., 3.]]).cuda()

#in1_mul = torch.tensor([[3., 1., 2.], [6., 2., 4.], [12., 4., 8.]]).cuda()
#in1 = [torch.tensor([[1., 2., 4.], [3., 1., 2.]])]

in1_mul = torch.tensor([[11., 5., 11.], [10., 7., 8.], [13., 9., 9.]]).cuda()
in1 = [torch.tensor([[3., 1., 1.], [2., 0., 2.]]), torch.tensor([[0., 2.], [3., 1.]]), torch.tensor([[5.], [1.]])]

print("input:", in1)
print("input multiplied out for old ReLU:", in1_mul)
result = module(in1_mul)
print("old ReLU Result: ", result)
print(f"\tmin: {torch.min(result)} max: {torch.max(result)}")

result = relu(in1, 2, [4, 1, 2])
a_max, a_min = find_range(result, 2)
print("new ReLU Result: ", result)
print(f"\tmin: {a_min} max: {a_max}")
