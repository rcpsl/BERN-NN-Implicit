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
from Bern_NN import NetworkModule as NetworkModuleEBF, bern_coeffs_inputs
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
from bern_to_poly import multi_bern_to_pow_poly

@dataclass
class Results:
    bern_runtime: float
    bern_vol: float
    bern_ibf_runtime: float
    bern_ibf_vol: float

def run_experiment(n_vars: int,
                   n_layers: int,
                   hidden_dim: int,
                   output_dim: int,
                   intervals: List[List[float]],
                   lin_itr_numb: int,
                   assume_quadrant: bool,
                   rank,
                   size,
                   device):
    assert rank < size
    assert dist_is_used()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    conv = multi_bern_to_pow_poly(dims=n_vars, intervals=intervals)
    torch_intervals = torch.tensor(intervals, dtype=torch.float32).to(device)

    network_size, network_weights, network_biases = build_network(
            n_vars, n_layers, hidden_dim, output_dim, device)

    
    print ('starting bern-nn')
    ### BERN-NN
    if rank == 0:
        inputs = bern_coeffs_inputs(n_vars, intervals)
        inputs = torch.stack(inputs)
        network_moduleEBF = NetworkModuleEBF(n_vars,
                                             torch_intervals,
                                             network_weights,
                                             network_biases,
                                             network_size,
                                             2,
                                             lin_itr_numb)
        start_time = time.time()
        with torch.no_grad():
            res_under, res_over, network_nodes_des = network_moduleEBF(inputs)
        bern_runtime = time.time() - start_time
        bern_vol = conv.Bern_NN_volume(res_under, res_over).cpu()
    else:
        bern_runtime, bern_vol = None, None

    torch.distributed.barrier()

    print ('starting bern-nn-ibf')
    ### IBF
    inputs = generate_inputs(n_vars, torch_intervals, device=device)
    with torch.no_grad():
        network = NetworkModule(n_vars,
                                torch_intervals,
                                network_weights,
                                network_biases,
                                network_size,
                                lin_itr_numb,
                                assume_quadrant,
                                device=device)
        time_start = time.time()
        res_under, res_over, under_degrees, over_degrees = network(inputs, rank, size)
        bern_ibf_runtime = time.time() - time_start
    bern_ibf_vol = conv.Bern_NN_IBF_volume(res_under, res_over, under_degrees, over_degrees).cpu()

    torch.distributed.barrier()

    return Results(bern_runtime=bern_runtime,
                   bern_vol=bern_vol,
                   bern_ibf_runtime=bern_ibf_runtime,
                   bern_ibf_vol=bern_ibf_vol)
