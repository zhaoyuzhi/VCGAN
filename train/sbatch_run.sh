#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p pixel
#SBATCH --gres=gpu:4
#SBATCH --nodelist=HK-IDC1-10-1-75-52
#SBATCH --job-name=second
#SBATCH -o ./logs/%j.txt
srun --mpi=pmi2 sh second.sh
