#!/bin/bash
#SBATCH --mem=30g
#SBATCH -c 15
#SBATCH --gres=gpu:4
#SBATCH --time=5-00
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

module load torch
module load tensorflow
module load nvidia

dir=/cs/labs/daphna/idan.azuri/PyTorch-GAN-Experiments-Env

cd $dir
source /cs/labs/daphna/idan.azuri/venv3.6/bin/activate


python3 main.py --config miniimagenet_config
