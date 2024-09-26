import onnx
import onnx2pytorch
import os
import torch.nn as nn
import csv

def get_acas_weights(path_to_model):
    """
        Args:
            path_to_model: path to ONNX files
        Return:
            params: list of pairs of models params
    """
    try:
        onnx_model = onnx.load(path_to_model) 
        pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
        modules = list(pytorch_model.modules())[1:]
        model = nn.Sequential(*modules[2:])
        weights = []
        biases = []
        for module in modules:
            module_name = str(module).split('(')[0]
            if 'Linear' in module_name:
                weights.append(module.weight.T)
                biases.append(module.bias)
        # print('model', model)
        return (model, weights, biases)
    except Exception as e:
        print(e)


# Revising the code to handle different data formats
def process_csv(file_path):
    data_dict = {}

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                # Split the second column path to extract prev and tau
                path_elements = row[1].split('/')
                onnx_file = path_elements[-1]  # The last part of the path
                onnx_parts = onnx_file.split('_')
                prev = int(onnx_parts[2])
                tau = int(onnx_parts[3])

                # Split the third column path to extract spec
                vnnlib_file = row[2].split('/')[-1]  # The last part of the path
                spec = int(vnnlib_file.split('_')[1].split('.')[0])

                # Add to the dictionary
                key = (prev, tau, spec)
                data_dict[key] = row[4]
            except IndexError:
                # Handle rows that don't match the expected format
                continue

    return data_dict


def cumulative_sum_sorted(L):
    # Sort the list in ascending order
    L.sort()

    # replac any value of 10000 in L with 116
    L = [116 if x == 10000 else x for x in L]

    # Initialize the result list with the first element
    res = [L[0]]

    # Iterate over the sorted list and add cumulative sums
    for i in range(1, len(L)):
        res.append(res[i-1] + L[i])

    return res



