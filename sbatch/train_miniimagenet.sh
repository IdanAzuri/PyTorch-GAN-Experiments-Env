#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 10
#SBATCH --gres=gpu:2
#SBATCH --time=2-20
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=0-5%6


module load torch
module load tensorflow
module load nvidia

dir=/cs/snapless/daphna/idan.azuri/PyTorch-GAN-Experiments-Env

cd $dir
source /cs/labs/daphna/idan.azuri/venv3.6/bin/activate


if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
python3 main.py --config pretrained_resnet50
fi

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
python3 main.py --config pretrained_resnet50_aug
fi

if [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
python3 main.py --config resnet50
fi

if [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
python3 main.py --config resnet50_aug
fi

if [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
python3 main.py --config 4conv
fi

if [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
python3 main.py --config 4conv_aug
fi

#python3 main.py --config pretrained_vgg19_bn_${SLURM_ARRAY_TASK_ID}
#python3 main.py --config pretrained_resnet50_${SLURM_ARRAY_TASK_ID}
