import torch
import itertools
import numpy as np

from copy import deepcopy

def get_multinomial_coef(power, terms):
    """
    calculates multinomial coefficients by
    power choose (terms[0], terms[1], ...)
    """
    denom = 0
    for term in terms:
        denom += torch.lgamma(term+1)
    return torch.exp(torch.lgamma(power+1) - denom)

def generate_multinomial_terms(power, n_terms):
    """
    generator that yields powers of each variable for terms in multinomial expansion
    ex. [1, 2, 1] = x1 * x2^2 * x3
    power: power entire expression is being raised to
    n_terms: number of terms in original expression
    """
    possible_values = torch.arange(power+1)
    #for val in torch.cartesian_prod(*[possible_values for _ in range(n_terms)]):
    for val in itertools.product(possible_values, repeat=n_terms):
        val = torch.tensor(val)
        if torch.sum(val) == power:
            if not val.shape:
                val = val.unsqueeze(0)
            yield val

def separated_poly_multiplication(polyA, polyB, n_vars):
    """
    multiplies two polynomials in separated list format together
    """
    if polyA.numel() == 1:
        result = polyB.clone()
        result[0] *= polyA.item()
        return result
    if polyB.numel() == 1:
        result = polyA.clone()
        result[0] *= polyB.item()
        return result
    if not torch.any(polyA) or not torch.any(polyB):
        return torch.zeros((n_vars, 1), device=polyA.device)
    padding = polyB.shape[1] - 1
    polyB = polyB.flip(1).reshape(n_vars, 1, -1)
    return torch.nn.functional.conv1d(polyA, polyB, padding=padding, groups=n_vars)

def poly_power(pA, n_vars, power):
    """
    calculates pA raised to the power power
    """
    if power == 0:
        return torch.ones((n_vars, 1), device=pA.device)
    if power == 1:
        return pA.clone()
    padding = pA.shape[1] - 1
    kernel = pA.flip(1).reshape(n_vars, 1, -1)
    for i in range(power-1):
        pA = torch.nn.functional.conv1d(pA, kernel, padding=padding, groups=n_vars)
    return pA

def multiply_polynomial_list(l, n_vars):
    """
    multiply all polynomials in the list together
    """
    if len(l) == 1:
        return l[0].clone() # TODO: Probably better way to copy
    result = separated_poly_multiplication(l[0], l[1], n_vars)
    for i in range(2, len(l)):
        result = separated_poly_multiplication(result, l[i], n_vars)
    return result

def calculate_relu_term(terms, n_vars, power, relu_coef=1):
    """
    calculates one term of the relu, given terms of expression, power of term, 
    and coefficient
    terms: list of tensors
    n_vars: number of variables, shape[0]
    power: power term is raised to
    """
    if power == 0:
        result = torch.ones((n_vars, 1), device=terms[0].device)
        result[0] = relu_coef
        return [result]
    results = []
    for multi_coeffs in generate_multinomial_terms(power, len(terms)):
        mult_coeff = get_multinomial_coef(torch.tensor(power), multi_coeffs)
        intermediates = []
        for term, term_power in zip(terms, multi_coeffs):
            intermediates.append(poly_power(term, n_vars, term_power))
        result = multiply_polynomial_list(intermediates, n_vars)
        result[0, :] *= mult_coeff * relu_coef
        results.append(result)
    return results

def relu(terms, n_vars, coefficients):
    """
    calculates result of feeding terms through relu function
    """
    if not torch.any(coefficients):
        return [torch.zeros((n_vars, 1))]
    terms = remove_all_zero_terms(terms)
    terms = combine_constants(terms, n_vars)
    scaled_terms = bern_scaling_list(terms)
    results = []
    for power, coef in enumerate(coefficients):
        if coef != 0:
            intermediate = calculate_relu_term(scaled_terms, n_vars, power, coef)
            #results.append(intermediate)
            results += intermediate
    results = bern_unscaling_list(results)
    results = combine_constants(results, n_vars)
    #print(results)
    return results


def one_dim_coeffs(degree):
    """
    generates multinomial coefficients as a tensor in 1D
    degree: degree for multinomial coefficients
    """
    N = torch.tensor(degree)
    R = torch.arange(degree + 1)
    a = torch.lgamma(N + 1)
    b = torch.lgamma(N-R+1)
    c = torch.lgamma(R+1)
    return torch.exp(a-b-c)

def degree_elevation(pA, n_vars, old_degree, new_degree):
    """
    performs degree elevation on pA from old degree to new degree
    """
    if old_degree == new_degree:
        return pA.clone()
    nCr_diff = one_dim_coeffs(new_degree - old_degree).reshape(1, 1, -1)
    pA_reshaped = pA.reshape(n_vars, 1, -1)
    # multinomial coefficients are symmetrical about axis 0 so no flip needed?
    return torch.nn.functional.conv1d(pA_reshaped, nCr_diff, padding=new_degree - old_degree).reshape(n_vars, -1)

def remove_all_zero_terms(l):
    new_l = []
    for t in l:
        if torch.any(t):
            new_l.append(t)
    return new_l

def combine_constants(l, n_vars):
    new_l = []
    constant = 0
    for t in l:
        if t.numel() == n_vars:
            constant += t[0,0]
        else:
            new_l.append(t)
    new_const = torch.ones((n_vars, 1), device=l[0].device)
    new_const[0,0] = constant
    new_l.append(new_const)
    return new_l

def degree_elevation_list(l, n_vars, new_degree):
    new_l = []
    for t in l:
        new_l.append(degree_elevation(t, n_vars, t.shape[1]-1, new_degree))
    return new_l

def bern_scaling(t):
    nCr_coeffs = one_dim_coeffs(t.shape[1] - 1)
    return t * nCr_coeffs

def bern_unscaling(t):
    nCr_coeffs = one_dim_coeffs(t.shape[1]-1)
    return t / nCr_coeffs

def bern_scaling_list(l):
    new_l = []
    for i in range(len(l)):
        new_l.append(bern_scaling(l[i]))
    return new_l

def bern_scaling_list_mut(l):
    for i in range(len(l)):
        l[i] = bern_scaling(l[i])
    return l

def bern_unscaling_list(l):
    new_l = []
    for i in range(len(l)):
        new_l.append(bern_unscaling(l[i]))
    return new_l

def bern_unscaling_list_mut(l):
    for i in range(len(l)):
        l[i] = bern_unscaling(l[i])
    return l

def combine(t, n_vars):
    #print("COMBINE", t)
    max_power = max(x.shape[1] for x in t) - 1
    pre_degree_elev_scaled = bern_scaling_list(t)
    degree_elev = degree_elevation_list(pre_degree_elev_scaled, n_vars, max_power)
    t = bern_unscaling_list(degree_elev)
    if n_vars == 2:
        return sum(torch.outer(x[0], x[1]) for x in t)
    return sum(x[0] for x in t)

def find_range(t, n_vars):
    """
    find the max and min coefficients given a list of terms
    t: list of tensors, that are all the same size
    n_vars: number of variables
    """
    copy_t = deepcopy(t)
    max_power = max(x.shape[1] for x in t) - 1
    pre_degree_elev_scaled = bern_scaling_list(t)
    degree_elev = degree_elevation_list(pre_degree_elev_scaled, n_vars, max_power)
    new_t = bern_unscaling_list(degree_elev)
    curr_max = -np.Inf
    curr_min = np.Inf

    for powers in itertools.product(range(max_power+1), repeat=n_vars):
        current_value = 0.
        for term in new_t:
            prod = 1
            for var, p in enumerate(powers):
                prod *= term[var, p]
            current_value += prod
        if current_value < curr_min:
            curr_min = current_value
        if current_value > curr_max:
            curr_max = current_value
    return curr_max, curr_min


if __name__ == "__main__":
    print("TEST get_multinomial_coef")
    result = get_multinomial_coef(torch.tensor(11), torch.tensor([1, 4, 4, 2]))
    if abs(result - 34650) > 1:
        print("\tFAILED\n")

    print("TEST generate_multinomial_terms")
    result = torch.stack([x for x in generate_multinomial_terms(3, 2)])
    expected = torch.tensor([[0,3],[1,2],[2,1],[3,0]])
    if not torch.allclose(result, expected):
        print("\tFAILED\n")

    print("TEST poly_power")
    result = poly_power(torch.tensor([[1, 2], [2, 4]], dtype=torch.float32), 2, 4)
    expected = torch.tensor([[1, 8, 24, 32, 16], [16, 128, 384, 512, 256]], dtype=torch.float32)
    if not torch.allclose(result, expected):
        print("\tFAILED\n")

    print("TEST separated_poly_multiplication")
    result = separated_poly_multiplication(torch.tensor([[1, 2], [2, 4]], dtype=torch.float32),
            torch.tensor([[3, 2], [2,1]], dtype=torch.float32), 2)
    expected = torch.tensor([[3, 8, 4], [4, 10, 4]], dtype=torch.float32)
    if not torch.allclose(result, expected):
        print("\tFAILED\n")

    print("TEST multiply_polynomial_list")
    l = [
            torch.tensor([[1., 2.], [1., 3.]]),
            torch.tensor([[2., 2.], [3., 3.]]),
            torch.tensor([[1., 1.], [0., 3.]]),
    ]
    result = multiply_polynomial_list(l, 2)
    expected = torch.tensor([[2, 8, 10, 4], [0, 9, 36, 27]], dtype=torch.float32)
    if not torch.allclose(result, expected):
        print("\tFAILED\n")

    print(calculate_relu_term([torch.tensor([[1., 2., 1.], [2., 4., 2.]])], 2, 2))
    print(relu([torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])], 3, [1, 2, 1]))

    a = [torch.tensor([[2., 3., 5.], [1., 2., 3.]]), torch.tensor([[2., 7.], [5., 1.]])]
    degree_elevated_a = degree_elevation_list(a, 2, 2)
    a_max, a_min = find_range(degree_elevated_a, 2)
    print("TEST degree_elevation + find max/min")
    print("Input", a)
    print("Degree Elevated", degree_elevated_a)
    print("max, min", a_max, a_min)

    a = [torch.tensor([[1., 2., 4.], [3., 1., 2.]])]
    relu_coeffs = [4, 1, 2]
    print(run_relu(a, 2, relu_coeffs))
