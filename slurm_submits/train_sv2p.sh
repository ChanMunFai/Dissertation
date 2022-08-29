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

# python main_sv2p.py --stage=3 --epochs=100 --learning_rate=1e-4 --beta_end=0.0001
# python main_sv2p.py --stage=3 --epochs=100 --learning_rate=1e-4 --beta_end=0.01
# python main_sv2p.py --stage=3 --epochs=100 --learning_rate=1e-4 --beta_end=0.1
# python main_sv2p.py --stage=3 --epochs=1000 --learning_rate=1e-6 --beta_end=0.001 --batch_size=52

### Train posterior networks 
# python -m sv2p.train_posterior_sv2p --epochs=50 --beta=10

### BouncingBall 50
# dset=BouncingBall_50
# python main_sv2p.py --stage=3 --dataset $dset --subdirectory=v1 --epochs=30 --batch_size=16 --wandb_on=True

# python3 main_sv2p.py --stage=2 --dataset=HealingMNIST_20 --subdirectory=v2 --epochs=50 --batch_size=32 --learning_rate=0.001 --wandb_on=True
python3 main_sv2p.py --stage=2 --dataset=DancingMNIST_20_v2 --subdirectory=v2 --epochs=50 --batch_size=32 --learning_rate=0.001 --wandb_on=True