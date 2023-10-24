import torch





###################################################
### repeat every term in TA tA - index times  
### n: number of variables
### tA: number of terms in TA
### degree_A: degree of pA
###################################################
def repeat_terms(TA, tA, n):
    # Create a tensor that represents the repeat pattern.
    repeat_counts = torch.arange(start=tA, end=0, step=-1, device=TA.device)  # [tA, tA-1, ..., 1]

    # Create a tensor of indices that represent the positions.
    indices = torch.arange(start=0, end=tA, device=TA.device)  # [0, 1, ..., tA-1]

    # Repeat each index (tA - index) times. This creates a pattern of indices.
    index_tensor = torch.repeat_interleave(indices, repeat_counts)

    # Adjust the indices to account for 'n' rows in each chunk.
    # Each index needs to be converted to 'n' row indices, resulting in a flat array of row indices.
    expanded_index_tensor = index_tensor * n
    row_indices = expanded_index_tensor.view(-1, 1).repeat(1, n) + torch.arange(n, device=TA.device)

    # Flatten the row_indices for gathering operation.
    flat_row_indices = row_indices.view(-1)

    # Gather the rows from the original tensor.
    result_tensor = torch.index_select(TA, 0, flat_row_indices)

    return result_tensor