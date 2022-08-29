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

# ### BouncingBall 20 - New Loss Function 
# dset=BouncingBall_20
# mod=KVAE_mod
# python main_kvae.py --subdirectory=v1 --model $mod --scale=0.3 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=None

### BouncingBall 50 - New Loss Function 
# dset=BouncingBall_50
# mod=KVAE_mod
# python main_kvae.py --subdirectory=v1 --model $mod --scale=0.3 --epoch=1 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True
# python main_kvae.py --subdirectory=bonus --model $mod --scale=0.3 --epoch=100 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --lstm_layers=3 --z_dim=5 --K=7 --wandb_on=True

### Bouncing Ball 20
# dset=BouncingBall_20
# python main_kvae.py --subdirectory=v1/attempt2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=10 --wandb_on=True
# python main_kvae.py --subdirectory=v1/attempt2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True

### Bouncing Ball 20 - MLP 
# dset=BouncingBall_20
# python main_kvae.py --alpha=mlp --subdirectory=mlp --scale=0.3 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=10 --wandb_on=True

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

### BouncingBall 50 - 3 LSTM layer 
# dset=BouncingBall_50
# python main_kvae.py --subdirectory=v3/new1 --scale=0.3 --epoch=100 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --lstm_layers=3 --wandb_on=True
# python main_kvae.py --subdirectory=v2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --lstm_layers=2 --wandb_on=True
# python main_kvae.py --subdirectory=v2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=100 --lstm_layers=2 --wandb_on=True

### MovingMNIST 
# dset=MovingMNIST
# python main_kvae.py --subdirectory=v2 --scale=1000.0 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --z_dim=5 --K=7 --wandb_on=True
# python main_kvae.py --model=KVAE_mod --subdirectory=v2 --scale=1.0 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=10 --a_dim=128 --z_dim=8 --K=5 --wandb_on=True

### Train BouncingBall 50 with extra parameters 
# dset=BouncingBall_50
# python main_kvae.py --subdirectory=v1/bonus --scale=0.3 --epoch=100 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --lstm_layers=1 --z_dim=5 --K=7 --wandb_on=True
# python main_kvae.py --subdirectory=v2/bonus --scale=0.3 --epoch=100 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --lstm_layers=2 --z_dim=5 --K=7 --wandb_on=True

### Train HealingMNIST 
dset=HealingMNIST_20
mod=KVAE_mod
# python main_kvae.py --subdirectory=v2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --lstm_layers=1 --a_dim=16 --z_dim=16 --K=3 --wandb_on=True
# python main_kvae.py --subdirectory=v3 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --lstm_layers=1 --a_dim=16 --z_dim=8 --K=3 --wandb_on=True
# python main_kvae.py --subdirectory=v4 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --lstm_layers=1 --a_dim=16 --z_dim=4 --K=3 --wandb_on=True
