import torch
import torch.multiprocessing as mp
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

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # the device ID uses the LOCAL rank
    device = f'cuda:{local_rank()}'

    n_trials = 2
    n_vars = 2
    hidden_dim = 50
    output_dim = 1
    lin_itr_numb = 1

    results = {}

    for n_layers in [2, 4, 6, 8, 10]:
        intervals = [[-1, 1] for i in range(n_vars)]
        intervals = torch.tensor(intervals, dtype=torch.float32).to(device)
        results[n_layers] = []
        for seed in range(n_trials):
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
                                           rank=rank,
                                           size=size,
                                           device=device)
            results[n_layers].append(trial_results)
    
    print(results)

      
if __name__ == "__main__":
    initialize('nccl')
    torch.cuda.set_device(f'cuda:{local_rank()}')
    mp.set_start_method("spawn")
    
    print(f'{rank()} / {world_size()}')

    run(rank(), world_size())
