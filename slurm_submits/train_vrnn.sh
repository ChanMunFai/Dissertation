#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/mc821/Dissertation/out/%j.out

export PATH=/vol/bitbucket/mc821/videopred_venv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/mc821/Dissertation

# python3 main_vrnn.py --dataset=BouncingBall_50 --wandb_on=True --epochs=300 --subdirectory=v3 --beta=1 --hdim=50 --zdim=50
# python3 main_vrnn.py --dataset=DancingMNIST_20_v2 --wandb_on=True --epochs=300 --subdirectory=v3 --beta=1 --hdim=128 --zdim=64
# python3 main_vrnn.py --dataset=HealingMNIST_20 --wandb_on=True --epochs=200 --subdirectory=v1 --beta=1 --hdim=128 --zdim=64
python3 main_vrnn.py --dataset=DancingMNIST_20_v2 --wandb_on=True --epochs=200 --subdirectory=v1 --beta=1 --hdim=128 --zdim=64



