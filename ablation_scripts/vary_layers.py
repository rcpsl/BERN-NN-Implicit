import torch
import torch.multiprocessing as mp
import pathlib
import pickle
import os
import sys
sys.path.append('../')
from network_modules_ibf_distributed import (
    initialize,
    dist_is_used,
    rank,
    local_rank,
    world_size
)

from ablation_trial import run_experiment

def run(rank, size):
    assert rank < size
    assert dist_is_used()

    # the device ID uses the LOCAL rank
    device = f'cuda:{local_rank()}'

    n_trials = 3
    n_vars = 2
    hidden_dim = 5
    output_dim = 1
    lin_itr_numb = 0
    assume_quadrant = True

    results = {}
    results['n_trials'] = n_trials
    results['n_vars'] = n_vars
    results['hidden_dim'] = hidden_dim
    results['output_dim'] = output_dim
    results['lin_itr_numb'] = lin_itr_numb

    for n_layers in [1, 2, 3, 4, 5, 6, 7]:
        print(n_layers)
        intervals = [[0.1, 1] for i in range(n_vars)]
        intervals = torch.tensor(intervals, dtype=torch.float32).to(device)
        results[n_layers] = []
        for seed in range(n_trials + 1):
            # Each trial needs to use a different random seed
            # This seed is set per-process, but each process should
            # use the same seed.
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            trial_results = run_experiment(n_vars=n_vars,
                                           n_layers=n_layers,
                                           hidden_dim=hidden_dim,
                                           output_dim=output_dim,
                                           intervals=intervals,
                                           lin_itr_numb=lin_itr_numb,
                                           assume_quadrant=assume_quadrant,
                                           rank=rank,
                                           size=size,
                                           device=device)
            if seed > 0:
                results[n_layers].append(trial_results)
            if rank == 0:
                print(trial_results)
        torch.cuda.empty_cache()
    
    if rank == 0:
        jobid = os.environ['SLURM_JOBID']
        results_dir = pathlib.Path('./ablation_data/')
        results_dir.mkdir(parents=True, exist_ok=True)
        results_filename = results_dir / f'layers_{jobid}_gpus{world_size()}.pickle'
        with open(results_filename, 'wb') as handle:
            pickle.dump(results, handle)
      
if __name__ == "__main__":
    initialize('nccl')
    torch.cuda.set_device(f'cuda:{local_rank()}')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    mp.set_start_method("spawn")
    
    print(f'{rank()} / {world_size()}')

    run(rank(), world_size())
