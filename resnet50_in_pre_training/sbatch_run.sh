#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p Pixel
#SBATCH --gres=gpu:8
#SBATCH --nodelist=SH-IDC1-10-5-34-166
#SBATCH --job-name=resnet50_fc_in
#SBATCH -o ./logs/resnet50_fc_in.txt
srun --mpi=pmi2 sh ./run.sh
