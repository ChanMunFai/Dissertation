### Find best checkpoint of each model by plotting validation MSE over time 

import os 
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import torchvision

from kvae.modules import KvaeEncoder, Decoder64, DecoderSimple 
from kvae.elbo_loss import ELBO
from kvae.model_kvae import KalmanVAE
from data.MovingMNIST import MovingMNIST
from dataset.bouncing_ball.bouncing_data import BouncingBallDataLoader

from kvae.inference_kvae import * 

class Ex1_Args:
    subdirectory = "experiment_1"
    dataset = "BouncingBall_50"
    model = 'KVAE'
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    state_dict_directory = "saves/BouncingBall_50/kvae/v1/attempt2/scale=0.3/scheduler_step=20/"

class Ex2_Args:
    subdirectory = "experiment_2"
    dataset = "BouncingBall_50"
    model = 'KVAE'
    alpha = "rnn"
    lstm_layers = 2
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    state_dict_directory = "saves/BouncingBall_50/kvae/v2/scale=0.3/scheduler_step=20/"

class Ex3_Args:
    subdirectory = "experiment_3"
    dataset = "BouncingBall_50"
    model = 'KVAE'
    alpha = "rnn"
    lstm_layers = 3
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/BouncingBall_50/kvae/v3/new1/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_99.pth"

class Ex3_Args:
    subdirectory = "experiment_3"
    dataset = "BouncingBall_50"
    model = 'KVAE'
    alpha = "rnn"
    lstm_layers = 3
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    state_dict_directory = "saves/BouncingBall_50/kvae/v3/new1/scale=0.3/scheduler_step=20/"

def load_val_dataset(dataset, batch_size): 
    if dataset == "BouncingBall_50": 
        val_set = BouncingBallDataLoader('dataset/bouncing_ball/50/val')
        val_loader = torch.utils.data.DataLoader(
                    dataset=val_set, 
                    batch_size=batch_size, 
                    shuffle=False)
    else: 
        print("Invalid Dataset")
        return 

    data, target = next(iter(val_loader))
    data = (data - data.min()) / (data.max() - data.min())
    data = torch.where(data > 0.5, 1.0, 0.0)

    target = (target - target.min()) / (target.max() - target.min())
    target = torch.where(target > 0.5, 1.0, 0.0)

    return data, target 

def find_avg_over_n_steps(n, mse_list): 
    relevant_mse = mse_list[:n]
    avg_mse = np.mean(relevant_mse)

    return avg_mse

def find_best_ckpt_over_n_steps(n, collated_mse_list, pathlist): 
    # Collated MSE list is list of average MSEs for each checkpoint 
    best_mse = np.min(collated_mse_list)
    idx_best = np.argmin(collated_mse_list)
    path_best = str(pathlist[idx_best])
    _, epoch_best = find_file_epoch_name(path_best)
    print(f"Best Checkpoint for 1st {n} time steps: {epoch_best} \t Value: {best_mse}")

    return 

def find_file_epoch_name(path_name): 
    """ Need to make sure that this works in general as it depends on how i name my files.
    """
    path_split = path_name.split("/")
    filename = path_split[-1]

    epoch_name = filename.split("_")[-1]
    epoch_name = epoch_name.split(".")[0]

    return filename, epoch_name

if __name__ == "__main__": 
    data, target = load_val_dataset("BouncingBall_50", batch_size = 32)

    # args = Ex1_Args
    # args = Ex2_Args
    args = Ex3_Args

    directory_in_str = args.state_dict_directory

    pathlist = Path(directory_in_str).rglob('*.pth')
    pathlist = list(pathlist)

    ### Keep track of best MSEs
    mse10_list = []
    mse25_list = []
    mse50_list = []
    
    for path in tqdm(pathlist):
        path_in_str = str(path)
        filename, epoch_name = find_file_epoch_name(path_in_str)
        args.state_dict_path = path_in_str

        print(epoch_name)

        ### Calculate MSE losses 
        mse_val = calc_model_losses_over_time(data, target, 50, args)
        plt.plot(mse_val, label=f"{epoch_name}")

        ### Measure best MSE over n steps 
        mse10_list.append(find_avg_over_n_steps(10, mse_val))
        mse25_list.append(find_avg_over_n_steps(25, mse_val))
        mse50_list.append(find_avg_over_n_steps(50, mse_val))

    ### Best MSE 
    find_best_ckpt_over_n_steps(10, mse10_list, pathlist)
    find_best_ckpt_over_n_steps(25, mse25_list, pathlist)
    find_best_ckpt_over_n_steps(50, mse50_list, pathlist)

    ### Plot
    mse_black = calc_black_losses_over_time(target)
    mse_last_seen = calc_last_seen_losses_over_time(data, target)
    plt.plot(mse_black, label="Black")
    plt.plot(mse_last_seen, label = "Last Seen Frame")

    plt.title("Validation MSE between predicted and ground truth")
    plt.ylabel('MSE')
    plt.xlabel('Time')
    plt.xticks(np.arange(0, len(mse_val), 5))
    plt.legend(loc="upper left")

    output_dir = directory_in_str
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + f"val_loss_over_time.jpeg")
    plt.close('all')
        

