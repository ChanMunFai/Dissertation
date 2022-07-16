# Code is used to test weights of Kalman Filter during inference 
# Used to debug for weird artefacts during prediction 

import os 
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import torchvision
import pandas as pd
import bar_chart_race as bcr

from kvae.modules import KvaeEncoder, Decoder64, DecoderSimple 
from kvae.elbo_loss import ELBO
from kvae.model_kvae import KalmanVAE
from data.MovingMNIST import MovingMNIST
from dataset.bouncing_ball.bouncing_data import BouncingBallDataLoader

# state_dict_path = "saves/BouncingBall_50/kvae/v2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_80.pth" 
# state_dict_path = "saves/BouncingBall_20/kvae/v1/attempt2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_60.pth"
state_dict_path = "saves/BouncingBall_50/kvae/v1/attempt2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_80.pth"
parser = argparse.ArgumentParser()
parser.add_argument('--subdirectory', default="experiment_1", type=str)
parser.add_argument('--dataset', default = "BouncingBall_50", type = str, 
                help = "choose between [MovingMNIST, BouncingBall_20, BouncingBall_50]")
parser.add_argument('--model', default="KVAE", type=str)
parser.add_argument('--alpha', default="rnn", type=str, 
                    help = "choose between [mlp, rnn]")
parser.add_argument('--lstm_layers', default=1, type=int, 
                    help = "Number of LSTM layers. To be used only when alpha is 'rnn'.")
parser.add_argument('--x_dim', default=1, type=int)
parser.add_argument('--a_dim', default=2, type=int)
parser.add_argument('--z_dim', default=4, type=int)
parser.add_argument('--K', default=3, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--device', default="cpu", type=str)
parser.add_argument('--scale', default=0.3, type=float)

args = parser.parse_args()

### Load model 
kvae = KalmanVAE(args = args).to(args.device)
state_dict = torch.load(state_dict_path, map_location = args.device)
kvae.load_state_dict(state_dict)

### Load dataset 
if args.dataset == "MovingMNIST": 
    train_set = MovingMNIST(root='dataset/mnist', train=True, download=True)
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=False)

elif args.dataset == "BouncingBall_20": 
    train_set = BouncingBallDataLoader('dataset/bouncing_ball/20/train')
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set, 
                batch_size=args.batch_size, 
                shuffle=False)

elif args.dataset == "BouncingBall_50": 
    train_set = BouncingBallDataLoader('dataset/bouncing_ball/50/train')
    train_loader = torch.utils.data.DataLoader(
                dataset=train_set, 
                batch_size=args.batch_size, 
                shuffle=False)

data, target = next(iter(train_loader))
data = data.to(args.device)
data = (data - data.min()) / (data.max() - data.min())
data = torch.where(data > 0.5, 1.0, 0.0)

target = target.to(args.device)
target = (target - target.min()) / (target.max() - target.min())
target = torch.where(target > 0.5, 1.0, 0.0)

example_data = data[1].unsqueeze(0)
example_target = target[1].unsqueeze(0)

x_predicted, _, _, weights = kvae.predict(example_data, pred_len = 50, return_weights = True)
# print(weights)

# We have artefacts from time steps 8 to 14 for Experiment 1 on 1st frame 
# print("====> Problematic frames")
# print(weights[:, 7:13])

# print("=====> Normal frames")
# print(weights[:,:7])

# print(weights[:,14])

### Create interactive bar charts 

### Dataset must be in Pandas wide format
# Each row represents a time step 
# Each column holds value for a particular category 

weights_pd = pd.DataFrame(weights[0].numpy())
weights_pd.columns = [0, 1, 2]

bcr.bar_chart_race(
    df=weights_pd,
    filename='plots/weights_over_time.mp4',
    orientation='h',
    sort='desc',
    n_bars=3,
    fixed_order=True,
    fixed_max=True,
    steps_per_period=10,
    interpolate_period=False,
    label_bars=True,
    bar_size=.95,
    period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},
    period_fmt='Index value - {x:.2f}',
    period_length=500,
    figsize=(5, 3),
    dpi=144,
    cmap='dark12',
    title='Weights Over Time for 1 Example',
    title_size='',
    bar_label_size=7,
    tick_label_size=7,
    scale='linear',
    writer=None,
    fig=None,
    bar_kwargs={'alpha': .7},
    filter_column_colors=False)  

