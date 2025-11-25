#!/bin/bash
#SBATCH --job-name=taxa_diffusion
#SBATCH --time=6-22:40:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16


export MASTER_PORT=12340
export WORLD_SIZE2=14


echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_NTASKS="${SLURM_NTASKS}
echo "SLURM_PROCID="${SLURM_PROCID}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# module spider cuda
# module load cuda/12.3.0
module load miniconda3
conda info --envs
conda activate taxa_diffusion

echo "Starting accelerate..."
srun python3 train.py --config configs/ifcb_diffusion.yaml --launcher slurm --wandb
