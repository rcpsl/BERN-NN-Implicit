"""
Multi-GPU Ablation, varying the input dimension.
"""
from dataclasses import dataclass
import torch
import torch.distributed as dist
import multiprocessing as mp
import time
import sys
sys.path.append('../')
from network_modules_ibf_distributed import (
    build_network,
    generate_inputs,
    NodeModule,
    LayerModule,
    NetworkModule,
    initialize,
    dist_is_used,
    rank,
    local_rank,
    world_size
)

from typing import List

import ibf_minmax_cpp

@dataclass
class Results:
    runtime: float
    lower_bound: float
    upper_bound: float

def run_experiment(n_vars: int,
                   n_layers: int,
                   hidden_dim: int,
                   output_dim: int,
                   intervals: List[List[float]],
                   lin_itr_numb: int,
                   rank,
                   size,
                   device):
    assert rank < size
    assert dist_is_used()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    inputs = generate_inputs(n_vars, intervals, device=device)
    network_size, network_weights, network_biases = build_network(
            n_vars, n_layers, hidden_dim, output_dim, device)

    time_start = time.time()
    with torch.no_grad():
        network = NetworkModule(n_vars,
                                intervals,
                                network_weights,
                                network_biases,
                                network_size,
                                lin_itr_numb,
                                device=device)
        res_under, res_over = network(inputs, rank, size)
    time_end = time.time()
    
    # get the min of the lower bound and max of the upper bound
    res_under = torch.reshape(res_under[0], (res_under[0].size(0) // n_vars, n_vars, res_under[0].size(1)))
    res_over = torch.reshape(res_over[0], (res_over[0].size(0) // n_vars, n_vars, res_over[0].size(1)))
    l = ibf_minmax_cpp.ibf_minmax(res_under)[0]
    u = ibf_minmax_cpp.ibf_minmax(res_over)[1]

    return Results(runtime=time_end - time_start,
                   lower_bound=l,
                   upper_bound=u)
