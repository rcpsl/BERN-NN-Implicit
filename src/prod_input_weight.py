import torch
import ibf_utils


def ibf_list_prod_input_weights(inputs, weights):
    """
    Every weight is used to multiply the first index of every term
    """
    result = []
    for ten, weight in zip(inputs, weights):
        if weight != 0:
            for t in ten:
                t = torch.tensor(t).to(weights.device)
                term = t+ 0
                term[0,:] *= weight
                result.append(term)
    return result

def ibf_tensor_prod_input_weights(inputs, weights):
    """
    inputs is a set of polynomials in IBF format
    weights is the parameters for one neuron? 
        TODO: change to weights for a batch of neurons
    """
    assert inputs.dim() == 4, 'expects four tensor dims [polynomials, terms, vectors, vars]'
    assert inputs.size(0) == weights.size(0), 'expects one polynomial for each weight'

    # Repeat the weights for each term in the corresponding polynomial.
    rep = torch.repeat_interleave(weights, repeats=inputs.size(1) * inputs.size(3), dim=0)
    term_scale = rep.reshape(inputs.size(0), inputs.size(1), inputs.size(3))
    # Overwrite into a tensor of ones. 
    ones = torch.ones_like(inputs, device=inputs.device)
    ones[:,:,0,:] = term_scale

    # Element-wise product scales one vector of each term by corresponding weight.
    scaled = inputs * ones

    # Reshaping essentially concatenates all terms into one large polynomial.
    # This is the sum of the scaled polynomials
    return scaled.reshape(-1, inputs.size(2), inputs.size(3))


if __name__ == '__main__':
    weights = torch.tensor([2.0])

    ibf_list = [[
            torch.tensor([[1, 2, 3], [4, 5, 6]]).float(),
            torch.tensor([[1, 2],    [4, 5]]).float()
    ]]

    res_l = ibf_list_prod_input_weights(ibf_list, weights)
    print(res_l)
    #res1_t = ibf_utils.ibf_batch_to_tensor(res_l, 3)
    #print(res1_t)


    ibf_list = [
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2],    [4, 5]]
        ],
        [
            [[1, 1], [1, 1]],
            [[2, 2], [2, 2]]
        ]
    ]
    weights = torch.tensor([2.0, 3.0])

    ibf_t = ibf_utils.ibf_batch_to_tensor(ibf_list, 3)
    res_t = ibf_tensor_prod_input_weights(ibf_t, weights)

    print(res_t)


