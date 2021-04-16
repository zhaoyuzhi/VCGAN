#!/bin/bash
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0
#SBATCH -p Pixel
#SBATCH --gres=gpu:1
#SBATCH --nodelist=SH-IDC2-172-20-20-78
#SBATCH --job-name=psnr
#SBATCH -o ./logs/%j.txt
srun --mpi=pmi2 sh folders_videvo_sh.sh
