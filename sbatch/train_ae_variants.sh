#!/bin/bash
#SBATCH --mem=2g
#SBATCH -c 4
#SBATCH --gres=gpu:m60:1
#SBATCH --time=1-20
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=1-5%5


module load torch
module load tensorflow
module load nvidia

dir=/cs/snapless/daphna/idan.azuri/PyTorch-GAN-Experiments-Env

cd $dir
source /cs/labs/daphna/idan.azuri/venv3.6/bin/activate


python3 auto-encoder.py --config ae_denoising_${SLURM_ARRAY_TASK_ID}
