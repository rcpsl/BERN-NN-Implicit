#!/usr/bin/bash 
#SBATCH --job-name=dist-bern-nn
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00

# One node needs to be used as the "host" for the rendezvuoz
# system used by torch. This just gets a list of the hostnames
# used by the job, and selects the first one.
HOST_NODE_ADDR=$(scontrol show hostnames | head -n 1)
NNODES=$(scontrol show hostnames | wc -l)

module purge
module load anaconda/2022.05
module load cuda
source ~/.mycondaconf
conda activate bernstein-poly

torchrun_with_gpus () {
    # $1 is the number of nodes (with ntasks == nnodes)
    # $2 is the number of GPUs per node
    srun --nodes $1 --ntasks $1 torchrun \
        --nnodes $1 \
        --nproc_per_node $2 \
        --max_restarts 2 \
        --rdzv_backend c10d \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_endpoint $HOST_NODE_ADDR \
        --redirects 3 \
        --tee 3 \
        strong_scaling.py;
}

torchrun_with_gpus 1 1 
torchrun_with_gpus 1 2 
torchrun_with_gpus 1 4 
torchrun_with_gpus 2 4 
