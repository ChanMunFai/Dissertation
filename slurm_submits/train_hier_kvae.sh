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

### MinecraftRL 
# python3 main_hier_kvae.py --dataset=MinecraftRL --levels=1 --factor=1 --scale=3.0 --epochs=90 --z_dim=64 --learning_rate=1e-5 --scheduler_step=10 --subdirectory=v4 --batch_size=32 --wandb_on=True

# python3 main_hier_kvae.py --dataset=MinecraftRL --levels=2 --factor=1 --scale=0.3 --epochs=90 --z_dim=64 --learning_rate=0.007 --scheduler_step=10 --subdirectory=v3 --batch_size=32 --wandb_on=True
# python3 main_hier_kvae.py --dataset=MinecraftRL --levels=2 --factor=2 --scale=0.3 --epochs=90 --z_dim=64 --learning_rate=0.007 --scheduler_step=10 --subdirectory=v3 --batch_size=32 --wandb_on=True

# python3 main_hier_kvae.py --dataset=MinecraftRL --levels=3 --factor=1 --scale=2.0 --epochs=90 --z_dim=64 --learning_rate=0.007 --scheduler_step=10 --subdirectory=v3 --batch_size=32 --wandb_on=True
# python3 main_hier_kvae.py --dataset=MinecraftRL --levels=3 --factor=2 --scale=0.3 --epochs=90 --z_dim=64 --learning_rate=0.007 --scheduler_step=10 --subdirectory=v3 --batch_size=32 --wandb_on=True




### BouncingBall 20 
# dset=BouncingBall_20
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# dset=BouncingBall_20
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=128 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# dset=BouncingBall_50
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=4 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=6 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=1 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v3 --levels=2 --factor=1 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v3 --levels=2 --factor=2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v3 --levels=1 --factor=1 --scale=0.3 --epoch=1 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v1 --levels=3 --factor=1 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=3 --factor=2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

### Test v2
# dset=BouncingBall_50
# python main_hier_kvae.py --subdirectory=test --levels=3 --factor=2 --scale=0.3 --epoch=1 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# dset=HealingMNIST_20
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=1 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=3 --factor=1 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=3 --factor=2 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=4 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=6 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=10 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v1 --levels=1 --factor=1 --scale=0.3 --epoch=90 --dataset $dset --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

### HealingMNIST_5 # use a_dim = 24 instead for sharper images 
# dset=HealingMNIST_5

# python main_hier_kvae.py --subdirectory=v2 --levels=1 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=2 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=4 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=6 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=10 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v2 --levels=3 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=3 --factor=2 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=3 --factor=4 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

### Redo HealingMNIST_20
# dset=HealingMNIST_20
# python main_hier_kvae.py --subdirectory=v2 --levels=1 --factor=1 --scale=1.0 --epoch=90 --dataset $dset --a_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=1 --factor=1 --scale=1.0 --epoch=90 --dataset $dset --a_dim=24 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 


### Redo HealingMNIST_5 with more parameters 
# dset=HealingMNIST_5
# python main_hier_kvae.py --subdirectory=v3 --levels=1 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v3 --levels=2 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=2 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=4 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=6 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=2 --factor=10 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v3 --levels=3 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=3 --factor=2 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v2 --levels=3 --factor=4 --scale=1 --epoch=90 --dataset $dset --a_dim=24 --z_dim=4 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

### Redo HealingMNIST_20 with more parameters 
# dset=HealingMNIST_20
# python main_hier_kvae.py --subdirectory=v3 --levels=1 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v3 --levels=2 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v3 --levels=2 --factor=2 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v3 --levels=2 --factor=4 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v3 --levels=2 --factor=6 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v3 --levels=2 --factor=10 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v3 --levels=3 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v3 --levels=3 --factor=2 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v3 --levels=3 --factor=4 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

### DancingMNIST_20 
# dset=DancingMNIST_20_v2
# python main_hier_kvae.py --subdirectory=v1 --levels=1 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=2 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=4 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=6 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=2 --factor=10 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

# python main_hier_kvae.py --subdirectory=v1 --levels=3 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=3 --factor=2 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python main_hier_kvae.py --subdirectory=v1 --levels=3 --factor=4 --scale=1 --epoch=90 --dataset $dset --a_dim=32 --z_dim=16 --batch_size=32 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 

### MovingMNIST 
# dset=MovingMNIST
# python3 main_hier_kvae.py --subdirectory=v1 --levels=1 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=128 --z_dim=64 --batch_size=64 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python3 main_hier_kvae.py --subdirectory=v2 --levels=1 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=128 --z_dim=32 --batch_size=64 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python3 main_hier_kvae.py --subdirectory=v4 --levels=1 --factor=1 --scale=1 --epoch=90 --dataset $dset --a_dim=200 --z_dim=128 --batch_size=64 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 
# python3 main_hier_kvae.py --subdirectory=v6 --levels=1 --factor=1 --scale=1 --epoch=100 --dataset $dset --K=10 --a_dim=200 --z_dim=128 --batch_size=64 --learning_rate=0.007 --initial_epochs=10 --scheduler_step=20 --wandb_on=True 