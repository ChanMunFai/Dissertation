### Inference for 1 layer KVAE on HealingMNIST_20
# Specifically, I used KVAE mod instead of KVAE so there may be some small differences in results

# Interested in differences between z_dim --> how large should z_dim be 

import os 
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import torchvision

from kvae.modules import KvaeEncoder, Decoder64, DecoderSimple 
from hier_kvae.model_hier_kvae import HierKalmanVAE
from kvae.model_kvae import KalmanVAE
from kvae.model_kvae_mod import KalmanVAEMod
from dataloader.moving_mnist import MovingMNISTDataLoader
from dataloader.bouncing_ball import BouncingBallDataLoader
from dataloader.healing_mnist import HealingMNISTDataLoader

from hier_kvae.inference_hier_kvae import plot_predictions, plot_predictions_diff_colours, plot_predictions_overlap, plot_reconstructions, calc_black_losses_over_time, calc_last_seen_losses_over_time, calc_model_losses_over_time

def load_dataset(dataset, batch_size): 
    if dataset == "MovingMNIST": 
        train_set = MovingMNISTDataLoader(root='dataset/mnist', train=True, download=True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=False)

    elif dataset == "BouncingBall_20": 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/20/train')
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=False)

    elif dataset == "BouncingBall_50": 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/50/train')
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=False)

    elif dataset == "HealingMNIST_20": 
        train_set = HealingMNISTDataLoader('dataset/HealingMNIST/20/', train = True, seen_len = 20)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=True)

    elif dataset == "HealingMNIST_50": 
        train_set = HealingMNISTDataLoader('dataset/HealingMNIST/50/', train = True, seen_len = 20)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=True)

    elif dataset == "HealingMNIST_5": 
        train_set = HealingMNISTDataLoader('dataset/HealingMNIST/5/', train = True, seen_len = 5)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=True)
    else: 
        raise NotImplementedError

    data, target = next(iter(train_loader))
    data = (data - data.min()) / (data.max() - data.min())
    data = torch.where(data > 0.5, 1.0, 0.0)
    target = (target - target.min()) / (target.max() - target.min())
    target = torch.where(target > 0.5, 1.0, 0.0)

    return data, target 

class args_a16_z8:
    subdirectory = "levels=1/adim=16_zdim=8"
    dataset = "HealingMNIST_20"
    model = 'KVAE_mod'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 16
    z_dim = 8
    K = 3
    levels = 1
    factor = 1
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/HealingMNIST_20/kvae_mod/v3/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_89.pth"

class args_a16_z4:
    subdirectory = "levels=1/adim=16_zdim=4"
    dataset = "HealingMNIST_20"
    model = 'KVAE_mod'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 16
    z_dim = 4
    K = 3
    levels = 1
    factor = 1
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/HealingMNIST_20/kvae_mod/v4/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_89.pth"


if __name__ == "__main__": 
    args_a16_z8 =args_a16_z8
    args_a16_z4 = args_a16_z4
    
    data, target = load_dataset("HealingMNIST_20", batch_size = 32)
    print(data.shape, target.shape)

    plot_predictions_overlap(data, target, 20, args_a16_z8)
    plot_predictions_diff_colours(data, target, 20, args_a16_z8)
    plot_predictions(data, target, 20, args_a16_z8)
    plot_reconstructions(data, args_a16_z8)

    plot_predictions_overlap(data, target, 20, args_a16_z4)
    plot_predictions_diff_colours(data, target, 20, args_a16_z4)
    plot_predictions(data, target, 20, args_a16_z4)
    plot_reconstructions(data, args_a16_z4)



