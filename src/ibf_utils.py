import torch


def ibf_list_to_tensor(old_format, max_degree):
    r"""
    This function converts the list-y old_format
    to the new tensor format. It's just intended
    for testing.
    Note: for this function, everything is a list, which is easy
    to write out for tests.
    """
    num_terms = len(old_format)
    term_size = len(old_format[0])
    new_format = torch.zeros(num_terms, term_size, max_degree)
    for term_idx, term in enumerate(old_format):
        for var_idx, var in enumerate(term):
            new_format[term_idx, var_idx, :len(var)] = torch.tensor(var)
    return new_format

def ibf_batch_to_tensor(old_format, max_degree):
    return torch.stack([ibf_list_to_tensor(o, max_degree) for o in old_format], 0)

if __name__ == '__main__':
    old = [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [1, 0]]]
    new = ibf_list_to_tensor(old, 4)
    assert new.size(0) == len(old)
    assert new.size(2) == 4
    assert all(new[0, 0] == torch.tensor([1, 2, 3, 0]))
