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

### Bouncing Ball 20
# dset=BouncingBall_20
# python main_kvae.py --subdirectory=v1/attempt2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=10 --wandb_on=True
# python main_kvae.py --subdirectory=v1/attempt2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True

### BouncingBall 50 - 1 LSTM layer 
# dset=BouncingBall_50
# python main_kvae.py --subdirectory=v1/attempt2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=10 --wandb_on=True
# python main_kvae.py --subdirectory=v1/attempt2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True
# python main_kvae.py --subdirectory=v1/attempt2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=100 --wandb_on=True

### BouncingBall 50 - 2 LSTM layer 
# dset=BouncingBall_50
# python main_kvae.py --subdirectory=v2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=10 --lstm_layers=2 --wandb_on=True
# python main_kvae.py --subdirectory=v2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --lstm_layers=2 --wandb_on=True
# python main_kvae.py --subdirectory=v2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=100 --lstm_layers=2 --wandb_on=True

### MovingMNIST 
dset=MovingMNIST
# python main_kvae.py --subdirectory=v2 --scale=1000.0 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --z_dim=5 --K=7 --wandb_on=True
# python main_kvae.py --subdirectory=v1 --scale=0.3 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=10 --z_dim=5 --K=7 --wandb_on=True

### Train reconstruction only 
python main_kvae.py --subdirectory=v3 --train_reconstruction=True --scale=1 --epoch=40 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=0 --scheduler_step=5 --z_dim=5 --K=7 --wandb_on=True
