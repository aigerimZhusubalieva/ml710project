#!/bin/bash
#SBATCH --job-name=ddp_vgg16
#SBATCH --output=ddp_vgg16.%j.out
#SBATCH --error=ddp_vgg16.%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p ws-ia
#SBATCH --time=01:00:00

source activate ML710  # Replace with your actual environment

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

srun python train.py
