#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/mc821/Dissertation/out/%j.out

export PATH=/vol/bitbucket/mc821/videopred_venv/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=xterm #TERM=vt100 
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/mc821/Dissertation

# python3 main_hier_kvae.py --subdirectory=v5 --levels=1 --factor=1 --scale=1 --epoch=90 --dataset=DancingMNIST_20_v2 --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True
# python3 main_hier_kvae.py --subdirectory=v5 --levels=4 --factor=4 --scale=1 --epoch=90 --dataset=DancingMNIST_20_v2 --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=None

dset=BouncingBall_50
python3 main_hier_kvae.py --subdirectory=updated --levels=3 --factor=1 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --a_dim=2 --z_dim=4 --K=3 --wandb_on=True 
python3 main_hier_kvae.py --subdirectory=updated --levels=3 --factor=2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --a_dim=2 --z_dim=4 --K=3 --wandb_on=True 
