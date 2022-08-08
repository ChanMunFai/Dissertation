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
from utils import count_parameters

# We visualise weights for the following models: 
# 1. KVAE (mod) with K = 3
# 2. KVAE (bonus) with K = 7 
# 3. KVAE hierachical with 3 layers, K = 3

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
        train_set = HealingMNISTDataLoader('dataset/HealingMNIST/v1/', train = True)
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

def load_kvae(Args): 
    args = Args()
    if args.model == "Hier_KVAE" or args.model == "KVAE_Hier": 
        kvae = HierKalmanVAE(args = args).to(args.device)

    elif args.model == "KVAE_mod": 
        kvae = KalmanVAEMod(args = args).to(args.device)

    elif args.model == "KVAE": 
        kvae = KalmanVAE(args = args).to(args.device)

    else: 
        raise NotImplementedError

    state_dict = torch.load(args.state_dict_path, map_location = args.device)
    kvae.load_state_dict(state_dict)

    return kvae 

def call_args(Args): 
    args = Args()
    return args

def plot_weights(args, x): 
    """ Plot a line chart of weights over time for 1 particular example. 
    """
    kvae = load_kvae(args)

    # Find the weights 
    with torch.no_grad(): 
        a_sample, *_ = kvae._encode(x)
        output = kvae._interpolate_matrices(a_sample)
        weights = output[-1]

    weights = weights.detach().cpu().numpy()

    output_dir = f"plots/{args.dataset}/{args.model}/{args.subdirectory}"
    os.makedirs(output_dir, exist_ok = True)

    plt.plot(weights)
    plt.title(f"Weights over time for KVAE {args.subdirectory}.")
    plt.savefig(output_dir + "/weights.jpeg")

    sample = torchvision.utils.make_grid(x[0],x[0].size(0))

    plt.imsave(output_dir + f"/sample.jpeg",
                sample.cpu().permute(1, 2, 0).numpy())

    plt.close('all')

    return 

def print_parameters(args): 
    kvae = load_kvae(args)
    params = count_parameters(kvae)
    print(f"Number of Parameters for {args.subdirectory} is {params}.")

class hier_kvae3_args:
    subdirectory = "levels=3_factor=1" 
    dataset = "BouncingBall_50"
    model = 'KVAE_Hier' # or KVAE 
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    levels = 3
    factor = 1
    state_dict_path = "saves/BouncingBall_50/kvae_hier/v1/levels=3/factor=1/kvae_state_dict_scale_89.pth"

class hier_kvae1_args:
    subdirectory = "levels=1_K=3" 
    dataset = "BouncingBall_50"
    model = 'KVAE_Hier'  
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    levels = 1
    factor = 1
    state_dict_path = "saves/BouncingBall_50/kvae_hier/v1/levels=1/factor=1/kvae_state_dict_scale_89.pth"

class kvae_bonus_args:
    subdirectory = "levels=1_K=7" 
    dataset = "BouncingBall_50"
    model = 'KVAE_mod'  
    alpha = "rnn"
    lstm_layers = 3
    x_dim = 1
    a_dim = 2
    z_dim = 5
    K = 7
    device = "cpu"
    scale = 0.3
    levels = 1
    factor = 1
    state_dict_path = "saves/BouncingBall_50/kvae_mod/bonus/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_99.pth"

if __name__ == "__main__": 
    data, target = load_dataset("BouncingBall_50", 1)
    x = data[0].unsqueeze(0)

    args_hier_kvae3 = hier_kvae3_args
    args_hier_kvae1 = hier_kvae1_args
    args_kvae_bonus = kvae_bonus_args

    print_parameters(args_hier_kvae3) # 72424
    print_parameters(args_kvae_bonus) # 113361
    print_parameters(args_hier_kvae1) # 72232

    # plot_weights(args_hier_kvae1, x)
    # plot_weights(args_hier_kvae3, x)
    # plot_weights(args_kvae_bonus, x)