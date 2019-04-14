#!/bin/bash
#SBATCH --mem=6g
#SBATCH -c 8
#SBATCH --gres=gpu:m60:1
#SBATCH --time=1-20
#SBATCH --mail-user=idan.azuri@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --array=0-4%5


module load torch
module load tensorflow
module load nvidia

dir=/cs/snapless/daphna/idan.azuri/PyTorch-GAN-Experiments-Env

cd $dir
source /cs/labs/daphna/idan.azuri/venv3.6/bin/activate


if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
echo $SLURM_ARRAY_TASK_ID

python3 main.py --config 4conv --mode predict
fi

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
echo $SLURM_ARRAY_TASK_ID
    python3 main.py --config resnet50 --mode predict
fi

if [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
echo $SLURM_ARRAY_TASK_ID

python3 main.py --config pretrained_resnet50_aug --mode predict
fi

if [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
echo $SLURM_ARRAY_TASK_ID

python3 main.py --config pretrained_vgg19_bn_2 --mode predict
fi

if [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
echo $SLURM_ARRAY_TASK_ID

python3 main.py --config pretrained_vgg19_bn_2_aug --mode predict
fi

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
echo $SLURM_ARRAY_TASK_ID

python3 main.py --config 4conv_aug --mode predict
fi

# if [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
# python3 main.py --config pretrained_resnet50 --mode predict
# fi



#python3 main.py --config pretrained_vgg19_bn_${SLURM_ARRAY_TASK_ID}
#python3 main.py --config pretrained_resnet50_${SLURM_ARRAY_TASK_ID}
