# Inference on DancingMNIST_20 

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
from dataloader.dancing_mnist import DancingMNISTDataLoader

from hier_kvae.inference_hier_kvae import plot_predictions, plot_predictions_diff_colours, plot_predictions_overlap, plot_reconstructions, calc_black_losses_over_time, calc_last_seen_losses_over_time, calc_model_losses_over_time

def load_dataset(dataset, batch_size): 
    if dataset == "DancingMNIST_20_v2": 
        train_set = DancingMNISTDataLoader('dataset/DancingMNIST/20/v2/', train = True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=False)

    elif dataset == "DancingMNIST_50_v2": 
        train_set = DancingMNISTDataLoader('dataset/DancingMNIST/50/v2/', train = True, seen_len = 20)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=False)

    elif dataset == "DancingMNIST_100_v2": 
        train_set = DancingMNISTDataLoader('dataset/DancingMNIST/100/v2/', train = True, seen_len = 20)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=False)

    else: 
        raise NotImplementedError

    data, target = next(iter(train_loader))
    data = (data - data.min()) / (data.max() - data.min())
    data = torch.where(data > 0.5, 1.0, 0.0)
    target = (target - target.min()) / (target.max() - target.min())
    target = torch.where(target > 0.5, 1.0, 0.0)

    return data, target 

class args_1_1:
    subdirectory = "levels=1_factor=1"
    dataset = "DancingMNIST_20_v2"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 32
    z_dim = 16
    K = 3
    levels = 1
    factor = 1
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v1/levels=1/factor=1/kvae_state_dict_scale_89.pth"

class args_2_1:
    subdirectory = "levels=2_factor=1"
    dataset = "DancingMNIST_20_v2"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 32
    z_dim = 16
    K = 3
    levels = 2
    factor = 1
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v1/levels=2/factor=1/kvae_state_dict_scale_89.pth"

class args_2_2:
    subdirectory = "levels=2_factor=2"
    dataset = "DancingMNIST_20_v2"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 32
    z_dim = 16
    K = 3
    levels = 2
    factor = 2
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v1/levels=2/factor=2/kvae_state_dict_scale_89.pth"

class args_2_4:
    subdirectory = "levels=2_factor=4"
    dataset = "DancingMNIST_20_v2"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 32
    z_dim = 16
    K = 3
    levels = 2
    factor = 2
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v1/levels=2/factor=4/kvae_state_dict_scale_89.pth"


class args_3_1:
    subdirectory = "levels=3_factor=1"
    dataset = "DancingMNIST_20_v2"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 32
    z_dim = 16
    K = 3
    levels = 3
    factor = 1
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v1/levels=3/factor=1/kvae_state_dict_scale_89.pth"

class args_3_2:
    subdirectory = "levels=3_factor=4"
    dataset = "DancingMNIST_20_v2"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 32
    z_dim = 16
    K = 3
    levels = 3
    factor = 2
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v1/levels=3/factor=2/kvae_state_dict_scale_89.pth"

class args_3_4:
    subdirectory = "levels=3_factor=4"
    dataset = "DancingMNIST_20_v2"
    model = 'Hier_KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 32
    z_dim = 16
    K = 3
    levels = 3
    factor = 4
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v1/levels=3/factor=4/kvae_state_dict_scale_89.pth"


if __name__ == "__main__": 
    data, target = load_dataset("DancingMNIST_20_v2", batch_size = 32)
    # data, target = load_dataset("DancingMNIST_50_v2", batch_size = 32)
    # data, target = load_dataset("DancingMNIST_100_v2", batch_size = 32)
    print(data.shape, target.shape)

    plot_predictions_overlap(data, target, 20, args_1_1)
    plot_predictions(data, target, 20, args_1_1)
    plot_reconstructions(data, args_1_1)
    plot_predictions_diff_colours(data, target, 20, args_1_1)

    plot_predictions_overlap(data, target, 20, args_2_1)
    plot_predictions(data, target, 20, args_2_1)
    plot_reconstructions(data, args_2_1)
    plot_predictions_diff_colours(data, target, 20, args_2_1)

    plot_predictions_overlap(data, target, 20, args_3_4)
    plot_predictions(data, target, 20, args_3_4)
    plot_reconstructions(data, args_3_4)
    plot_predictions_diff_colours(data, target, 20, args_3_4)

    def plot_mse_dancingmnist20(): 
        ### MSE over time 
        mse_1_1 = calc_model_losses_over_time(data, target, 20, args_1_1)

        mse_2_1 = calc_model_losses_over_time(data, target, 20, args_2_1)
        mse_2_2 = calc_model_losses_over_time(data, target, 20, args_2_2)
        mse_2_4 = calc_model_losses_over_time(data, target, 20, args_2_4)
       
        mse_3_1 = calc_model_losses_over_time(data, target, 20, args_3_1)
        mse_3_2 = calc_model_losses_over_time(data, target, 20, args_3_2)
        mse_3_4 = calc_model_losses_over_time(data, target, 20, args_3_4)

        mse_black = calc_black_losses_over_time(target)
        mse_last_seen = calc_last_seen_losses_over_time(data, target)

        ### Plotting 
        plt.plot(mse_1_1, label="levels = 1, factor = 1", linestyle='dashed')
        
        plt.plot(mse_2_1, label="levels = 2, factor = 1")
        plt.plot(mse_2_2, label="levels = 2, factor = 2")
        plt.plot(mse_2_4, label="levels = 2, factor = 4")

        plt.plot(mse_3_1, label="levels = 3, factor = 1")
        plt.plot(mse_3_2, label="levels = 3, factor = 2")
        plt.plot(mse_3_4, label="levels = 3, factor = 4")

        plt.plot(mse_black, label="Black")
        plt.plot(mse_last_seen, label = "Last Seen Frame")

        plt.title("MSE between ground truth and predicted frame over time")
        plt.ylabel('MSE')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(mse_black), 20))
        plt.legend(loc="upper left")

        output_dir = f"plots/DancingMNIST_20_v2/KVAE_Hier/"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"KVAE_loss_b.jpeg")
        plt.close('all')

    def plot_mse_dancingmnist80(): 
        ### MSE over time 
        mse_1_1 = calc_model_losses_over_time(data, target, 80, args_1_1)

        mse_2_1 = calc_model_losses_over_time(data, target, 80, args_2_1)
        mse_2_2 = calc_model_losses_over_time(data, target, 80, args_2_2)
        mse_2_4 = calc_model_losses_over_time(data, target, 80, args_2_4)
       
        mse_3_1 = calc_model_losses_over_time(data, target, 80, args_3_1)
        mse_3_2 = calc_model_losses_over_time(data, target, 80, args_3_2)
        mse_3_4 = calc_model_losses_over_time(data, target, 80, args_3_4)

        mse_black = calc_black_losses_over_time(target)
        mse_last_seen = calc_last_seen_losses_over_time(data, target)

        ### Plotting 
        plt.plot(mse_1_1, label="levels = 1, factor = 1", linestyle='dashed')
        
        plt.plot(mse_2_1, label="levels = 2, factor = 1")
        plt.plot(mse_2_2, label="levels = 2, factor = 2")
        plt.plot(mse_2_4, label="levels = 2, factor = 4")

        plt.plot(mse_3_1, label="levels = 3, factor = 1")
        plt.plot(mse_3_2, label="levels = 3, factor = 2")
        plt.plot(mse_3_4, label="levels = 3, factor = 4")

        plt.title("MSE between ground truth and predicted frame over time")
        plt.ylabel('MSE')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(mse_black), 5))
        plt.legend(loc="upper left")

        output_dir = f"plots/DancingMNIST_20_v2/KVAE_Hier/"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"KVAE_80steps_a.jpeg")

        plt.plot(mse_black, label="Black")
        plt.plot(mse_last_seen, label = "Last Seen Frame")

        plt.savefig(output_dir + f"KVAE_80steps_b.jpeg")

        plt.close('all')

    def plot_mse_dancingmnist180(): 
        ### MSE over time 
        mse_1_1 = calc_model_losses_over_time(data, target, 180, args_1_1)

        mse_2_1 = calc_model_losses_over_time(data, target, 180, args_2_1)
        mse_2_2 = calc_model_losses_over_time(data, target, 180, args_2_2)
        mse_2_4 = calc_model_losses_over_time(data, target, 180, args_2_4)
       
        mse_3_1 = calc_model_losses_over_time(data, target, 180, args_3_1)
        mse_3_2 = calc_model_losses_over_time(data, target, 180, args_3_2)
        mse_3_4 = calc_model_losses_over_time(data, target, 180, args_3_4)

        mse_black = calc_black_losses_over_time(target)
        mse_last_seen = calc_last_seen_losses_over_time(data, target)

        ### Plotting 
        plt.plot(mse_1_1, label="levels = 1, factor = 1", linestyle='dashed')
        
        plt.plot(mse_2_1, label="levels = 2, factor = 1")
        plt.plot(mse_2_2, label="levels = 2, factor = 2")
        plt.plot(mse_2_4, label="levels = 2, factor = 4")

        plt.plot(mse_3_1, label="levels = 3, factor = 1")
        plt.plot(mse_3_2, label="levels = 3, factor = 2")
        plt.plot(mse_3_4, label="levels = 3, factor = 4")

        plt.title("MSE between ground truth and predicted frame over time")
        plt.ylabel('MSE')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(mse_black), 20))
        plt.legend(loc="upper left")

        output_dir = f"plots/DancingMNIST_20_v2/KVAE_Hier/"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"KVAE_180steps_a.jpeg")

        plt.plot(mse_black, label="Black")
        plt.plot(mse_last_seen, label = "Last Seen Frame")

        plt.savefig(output_dir + f"KVAE_180steps_b.jpeg")

        plt.close('all')

    # plot_mse_dancingmnist20()
    # plot_mse_dancingmnist80()
    # plot_mse_dancingmnist180()


