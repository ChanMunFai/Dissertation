#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=/vol/bitbucket/mc821/Dissertation/out/%j.out

export PATH=/vol/bitbucket/mc821/Dissertation/bin/:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=xterm #TERM=vt100 
/usr/bin/nvidia-smi
uptime

cd /vol/bitbucket/mc821/Dissertation

dset=BouncingBall_50
# V2 
# mod=KVAE_hier_V2
# python main_hier_kvae.py --model $mod --subdirectory=testing --levels=3 --factor=1 --scale=0.3 --epoch=1 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=0 --scheduler_step=20 --wandb_on=True 
# # python main_hier_kvae.py --model $mod --subdirectory=testing --levels=1 --factor=1 --scale=0.3 --epoch=1 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=0 --scheduler_step=20 --wandb_on=True 

# V1
mod=KVAE_hier
python main_hier_kvae.py --model $mod --subdirectory=testing --levels=3 --factor=2 --scale=0.3 --epoch=1 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=0 --scheduler_step=20 --wandb_on=True 
python main_hier_kvae.py --model $mod --subdirectory=testing --levels=1 --factor=1 --scale=0.3 --epoch=1 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=0 --scheduler_step=20 --wandb_on=True 

