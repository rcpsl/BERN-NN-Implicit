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




def repeat_terms_device(TA, tA, n, device):
    # Ensure TA is on the correct device
    TA = TA.to(device)

    # Create a tensor that represents the repeat pattern.
    repeat_counts = torch.arange(start=tA, end=0, step=-1, device=device)  # [tA, tA-1, ..., 1]

    # Create a tensor of indices that represent the positions.
    indices = torch.arange(start=0, end=tA, device=device)  # [0, 1, ..., tA-1]

    # Repeat each index (tA - index) times. This creates a pattern of indices.
    index_tensor = torch.repeat_interleave(indices, repeat_counts)

    # Adjust the indices to account for 'n' rows in each chunk.
    # Each index needs to be converted to 'n' row indices, resulting in a flat array of row indices.
    expanded_index_tensor = index_tensor * n
    row_indices = expanded_index_tensor.view(-1, 1).repeat(1, n) + torch.arange(n, device=device)

    # Flatten the row_indices for gathering operation.
    flat_row_indices = row_indices.view(-1)

    # Gather the rows from the original tensor.
    result_tensor = torch.index_select(TA, 0, flat_row_indices)

    return result_tensor






###################################################
### delete whole zero rows from a 2D tensor
### T: 2D tensor
###################################################

def remove_zero_rows(tensor):
    # Check if the tensor is a 2D tensor
    assert tensor.dim() == 2, "Input needs to be a 2D tensor"

    # Step 1: Identify Zero Rows and Columns

    # For rows, we sum along columns (dim=1) and check if the sum is zero.
    # 'keepdim=True' is used to ensure that sum keeps the number of dimensions unchanged.
    zero_rows = torch.sum(tensor, dim=1, keepdim=True) == 0

    # For columns, we sum along rows (dim=0) and check if the sum is zero.
    zero_columns = torch.sum(tensor, dim=0, keepdim=True) == 0

    # Step 2: Remove Zero Rows and Columns

    # We use '~' to get the inverse of the 'zero_rows' and 'zero_columns' masks.
    # Then, we use these inverse masks to index the original tensor, effectively selecting
    # only those rows and columns where not all elements are zeros.

    # Removing rows with all elements equal to zero
    tensor = tensor[~zero_rows.squeeze(1)]

    # # Removing columns with all elements equal to zero
    # # The transpose operation is used to make column removal similar to row removal.
    # tensor = tensor.transpose(0, 1)[~zero_columns.squeeze(0)].transpose(0, 1)

    return tensor



### 1) test the function repeat_terms
n = 3
n_cols = 3
tA = 3

### create a 2D tensor TA where it has tA * n rows and n_cols columns, where the ith submatrix of size n x n_cols is all equal to i
range_tensor = torch.arange(start=1, end=tA  + 1, dtype=torch.float32)  # each element i will fill the ith submatrix
# Step 2: Expand and repeat to form the final tensor
# First, we make 'range_tensor' have the shape (tA, 1, 1) so we can broadcast it
range_tensor = range_tensor.view(tA, 1, 1)
# Now, we expand 'range_tensor' to have the shape (tA, n, n_cols)
submatrices = range_tensor.expand(tA, n, n_cols)
# Finally, we reshape 'submatrices' to be a 2D tensor with shape (tA * n, n_cols)
TA = submatrices.reshape(tA * n, n_cols)
# print(TA)

### call the function repeat_terms
result_tensor = repeat_terms(TA, tA, n)
# print(result_tensor)

# print('#############################################')
### 2) test the other operation

TA_mod2 = [TA[i * n:].clone() for i in  range(tA)]
for i in range(len(TA_mod2)):
    chunk = TA_mod2[i]
    chunk[torch.arange(n, chunk.size(0), n)] *= 2
# print('TA_mod2', TA_mod2)
TA_mod2 = torch.cat(TA_mod2, dim=0)
# print('TA_mod2', TA_mod2)







# # Usage:
# # Create a 2D tensor 'T' where some rows and columns are entirely zero
# T = torch.tensor([
#     [0., 1., 0.],
#     [0., 0., 0.],
#     [2., 0., 0.]
# ])  # Example tensor

# # Call the function
# result_tensor = remove_zero_rows(T)

# print(result_tensor)








# class NodeModuleList(nn.Module):
#     def __init__(self, nodes):
#         super(NodeModuleList, self).__init__()  # Make sure to call the correct superclass initializer
#         self.module_list = nn.ModuleList(nodes)  # Use a proper ModuleList for storing the nodes.

#     def forward(self, inputs_for_nodes):
#         # Your forward method should be correctly recognized now. 
#         # Make sure the inputs are correctly unpacked if they're tuples or lists.
#         results = [node(*input) for node, input in zip(self.module_list, inputs_for_nodes)]
#         return results
    
#     def __iter__(self):
#         return iter(self.module_list)  # This makes your class iterable, delegating iteration to the internal ModuleList.
    
# ###################################################
# ### class LayerModule() takes as inputs:
# ### layer_inputs_under and layer_inputs_over and 
# ### and propagate them through the layer. 
# ### n_vars: number of variables
# ### degree: degree for layer_inputs_under and layer_inputs_over
# ### layer_weights: layer's weights where the ith row
# ### for the layer_weights corresponds to the weights 
# ### for the ith node for that layer.
# ### layer_biases: the biases for the layer
# ### layer_size: the number of nodes for the layer
# ###################################################
# class LayerModule(torch.nn.Module):

#     def __init__(self, n_vars, layer_weights, layer_biases, layer_size, activation):
#         super().__init__()
#         self.n_vars = n_vars
#         self.layer_size = layer_size
#         ### get the psoitive and negative weights from the layer_weights
#         self._layer_weights_pos = torch.nn.functional.relu(layer_weights)
#         self._layer_weights_neg = (-1) * torch.nn.functional.relu((-1) * layer_weights)
#         self.layer_biases = layer_biases
#         self.activation = activation
#         # ### call self.nodes objects
#         # self.nodes = [NodeModule(self.n_vars, self._layer_weights_pos[i, :], self._layer_weights_neg[i, :], self.layer_biases[i], self.activation) for i in range(self.layer_size)]

#         # Wrap the nodes in a custom container module.
#         self.nodes = NodeModuleList([NodeModule(self.n_vars, self._layer_weights_pos[i, :], self._layer_weights_neg[i, :], self.layer_biases[i], self.activation) for i in range(self.layer_size)])


#     def forward(self, layer_inputs_under, layer_inputs_over, layer_inputs_under_degrees, layer_inputs_over_degrees):

#         # Step 1: Create a list of inputs for each node.
#         inputs_for_nodes = [
#             [layer_inputs_under, layer_inputs_over, layer_inputs_under_degrees, layer_inputs_over_degrees]
#             for _ in self.nodes
#         ]
        
#          # Step 2: Distribute the computations across all available GPUs.
#         layer_results = nn.parallel.data_parallel(
#                     module=self.nodes,  # The module being parallelized (a ModuleList of NodeModules in this case)
#                     inputs=inputs_for_nodes,  # The inputs to the module, appropriately batched
#                     device_ids=None  # None implies all available GPUs
#                 )
#         ### return the results in results_under and results_over and their degrees
#         return layer_results






































