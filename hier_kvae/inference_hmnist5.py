# Separate file for inference on HealingMNIST_5 for easier organisation 

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
        train_set = HealingMNISTDataLoader('dataset/HealingMNIST/20/', train = True, seen_len = 5)
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

class args_2_1:
    subdirectory = "levels=2_factor=1_scale=1"
    dataset = "HealingMNIST_5"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 24
    z_dim = 4
    K = 3
    levels = 2
    factor = 1
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/HealingMNIST_5/kvae_hier/v2/levels=2/factor=1/kvae_state_dict_scale_89.pth"

class args_2_2:
    subdirectory = "levels=2_factor=2"
    dataset = "HealingMNIST_5"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 24
    z_dim = 4
    K = 3
    levels = 2
    factor = 2
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/HealingMNIST_5/kvae_hier/v1/levels=2/factor=2/kvae_state_dict_scale_89.pth"

class args_2_4:
    subdirectory = "levels=2_factor=2"
    dataset = "HealingMNIST_5"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 24
    z_dim = 4
    K = 3
    levels = 2
    factor = 4
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/HealingMNIST_5/kvae_hier/v1/levels=2/factor=4/kvae_state_dict_scale_89.pth"

class args_3_1:
    subdirectory = "levels=3_factor=1"
    dataset = "HealingMNIST_5"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 24
    z_dim = 4
    K = 3
    levels = 3
    factor = 1
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/HealingMNIST_5/kvae_hier/v1/levels=3/factor=1/kvae_state_dict_scale_89.pth"

class args_3_2:
    subdirectory = "levels=3_factor=1"
    dataset = "HealingMNIST_5"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 24
    z_dim = 4
    K = 3
    levels = 3
    factor = 2
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/HealingMNIST_5/kvae_hier/v1/levels=3/factor=2/kvae_state_dict_scale_89.pth"

class args_3_4:
    subdirectory = "levels=3_factor=4"
    dataset = "HealingMNIST_5"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 24
    z_dim = 4
    K = 3
    levels = 3
    factor = 4
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/HealingMNIST_5/kvae_hier/v1/levels=3/factor=4/kvae_state_dict_scale_89.pth"


if __name__ == "__main__": 
    args_2_1 = args_2_1 # level, factor 
    # args_2_2 = args_2_2
    # args_2_4 = args_2_4

    # args_3_1 = args_3_1 
    # args_3_2 = args_3_2
    # args_3_4 = args_3_4
    
    # data, target = load_dataset("HealingMNIST_20", batch_size = 32)
    data, target = load_dataset("HealingMNIST_5", batch_size = 32)
    print(data.shape, target.shape)

    plot_predictions_overlap(data, target, 5, args_2_1)
    plot_predictions_diff_colours(data, target, 5, args_2_1)
    plot_predictions(data, target, 5, args_2_1)

    # plot_predictions(data, target, 5, args_3_4)
    plot_reconstructions(data, args_2_1)

    def plot_mse_hmnist20(): 
        ### MSE over time 
        mse_2_1 = calc_model_losses_over_time(data, target, 20, args_2_1)
        mse_2_2 = calc_model_losses_over_time(data, target, 20, args_2_2)
        mse_2_4 = calc_model_losses_over_time(data, target, 20, args_2_4)

        mse_3_1 = calc_model_losses_over_time(data, target, 20, args_3_1)
        mse_3_2 = calc_model_losses_over_time(data, target, 20, args_3_2)

        mse_black = calc_black_losses_over_time(target)
        mse_last_seen = calc_last_seen_losses_over_time(data, target)

        ### Plotting 
        plt.plot(mse_2_1, label="levels = 2, factor = 1")
        plt.plot(mse_2_2, label="levels = 2, factor = 2")
        plt.plot(mse_2_4, label="levels = 2, factor = 4")

        plt.plot(mse_3_1, label="levels = 3, factor = 1")
        plt.plot(mse_3_2, label="levels = 3, factor = 2")

        plt.plot(mse_black, label="Black")
        plt.plot(mse_last_seen, label = "Last Seen Frame")

        plt.title("MSE between ground truth and predicted frame over time")
        plt.ylabel('MSE')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(mse_black), 5))
        plt.legend(loc="upper left")

        output_dir = f"plots/HealingMNIST_20/KVAE_Hier/"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"KVAE_loss.jpeg")
        plt.close('all')

    def plot_mse_hmnist50(): # but actually predict for 80 time steps
        
        mse_2_1 = calc_model_losses_over_time(data, target, 80, args_2_1)
        mse_2_2 = calc_model_losses_over_time(data, target, 80, args_2_2)
        mse_2_4 = calc_model_losses_over_time(data, target, 80, args_2_4)

        mse_3_1 = calc_model_losses_over_time(data, target, 80, args_3_1)
        mse_3_2 = calc_model_losses_over_time(data, target, 80, args_3_2)

        mse_black = calc_black_losses_over_time(target)
        mse_last_seen = calc_last_seen_losses_over_time(data, target)

        ### Plotting 
        plt.plot(mse_2_1, label="levels = 2, factor = 1")
        plt.plot(mse_2_2, label="levels = 2, factor = 2")
        plt.plot(mse_2_4, label="levels = 2, factor = 4")

        plt.plot(mse_3_1, label="levels = 3, factor = 1")
        plt.plot(mse_3_2, label="levels = 3, factor = 2")

        plt.plot(mse_black, label="Black")
        plt.plot(mse_last_seen, label = "Last Seen Frame")

        plt.title("MSE between ground truth and predicted frame over time")
        plt.ylabel('MSE')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(mse_black), 5))
        plt.legend(loc="lower left")

        output_dir = f"plots/HealingMNIST_50/KVAE_Hier/"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"KVAE_loss.jpeg")
        plt.close('all')

    # plot_mse_hmnist20()
    # plot_mse_hmnist50()

