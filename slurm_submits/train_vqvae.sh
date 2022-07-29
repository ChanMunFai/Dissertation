#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/mc821/Dissertation/out/%j.out

export PATH=/vol/bitbucket/mc821/Dissertation/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100  # TERM=xterm
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/mc821/Dissertation

python -m vqvae.main_vqvae --subdirectory=v1 --epochs=100 --batch_size=64 --wandb_on=True 

