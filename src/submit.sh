#!/usr/bin/bash 
#SBATCH --job-name=dist-bern-nn
#SBATCH -A amowli_lab_gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00

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

srun torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node 1 \
    --max_restarts 0 \
    --rdzv_backend c10d \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_endpoint $HOST_NODE_ADDR \
    --redirects 3 \
    --tee 3 \
    network_modules_ibf_distributed.py
