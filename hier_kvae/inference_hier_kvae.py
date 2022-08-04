# Test for updated predict function for 1 layer

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

from data.MovingMNIST import MovingMNIST
from dataset.bouncing_ball.bouncing_data import BouncingBallDataLoader

def load_dataset(dataset, batch_size): 
    if dataset == "MovingMNIST": 
        train_set = MovingMNIST(root='dataset/mnist', train=True, download=True)
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
    if args.model == "Hier_KVAE": 
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

def load_train_dataset(dataset, batch_size): 
    if dataset == "BouncingBall_50": 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/50/train')
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=False)
    else: 
        print("Invalid Dataset")
        return 

    data, target = next(iter(train_loader))
    data = (data - data.min()) / (data.max() - data.min())
    data = torch.where(data > 0.5, 1.0, 0.0)

    target = (target - target.min()) / (target.max() - target.min())
    target = torch.where(target > 0.5, 1.0, 0.0)

    return data, target 

def plot_predictions(x, target, pred_len, args):
    """ Plot predictions where ground truth and predictions are plotted 
    in different images (and directories)"""

    kvae = load_kvae(args)

    x_predicted = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/Hier_KVAE/{args.subdirectory}/predictions/"
        output_dir_gt = f"results/{args.dataset}/Hier_KVAE/{args.subdirectory}/ground_truth/"
        if not os.path.exists(output_dir_pred):
            os.makedirs(output_dir_pred)
        if not os.path.exists(output_dir_gt):
            os.makedirs(output_dir_gt)

        predicted_frames = torchvision.utils.make_grid(i,i.size(0))

        ground_truth = target[batch_item,:,:,:,:]
        ground_truth_frames = torchvision.utils.make_grid(ground_truth,ground_truth.size(0))
        stitched_frames = torchvision.utils.make_grid([ground_truth_frames, predicted_frames],1)

        plt.imsave(output_dir_pred + f"predictions_{batch_item}.jpeg",
                predicted_frames.cpu().permute(1, 2, 0).numpy())

        plt.imsave(output_dir_gt + f"ground_truth_{batch_item}.jpeg",
                ground_truth_frames.cpu().permute(1, 2, 0).numpy())
    return 

def plot_predictions_diff_colours(x, target, pred_len, args):
    """ Plot predictions and ground truth in the same image but with different colours.
    Ground truth - blue 
    Predictions - red
    Overlapped regions - blue + red = purple 
    """
    kvae = load_kvae(args)

    x_predicted = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/Hier_KVAE/{args.subdirectory}/Coloured_Predictions/"
        if not os.path.exists(output_dir_pred):
            os.makedirs(output_dir_pred)

        ground_truth = target[batch_item,:,:,:,:]
        empty_channel = torch.full_like(i, 0)
        stitched_video = torch.cat((i, empty_channel, ground_truth), 1)
        stitched_frames = torchvision.utils.make_grid(stitched_video, stitched_video.size(0))    
    
        plt.imsave(output_dir_pred + f"predictions_{batch_item}.jpeg",
                stitched_frames.cpu().permute(1, 2, 0).numpy())
    
    return 

def plot_predictions_overlap(x, target, pred_len, args):
    """ Plot overlaps of predictions and ground truth."""
    kvae = load_kvae(args)

    x_predicted = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/Hier_KVAE/{args.subdirectory}/Overlapped_Predictions/"
        if not os.path.exists(output_dir_pred):
            os.makedirs(output_dir_pred)

        ground_truth = target[batch_item,:,:,:,:]
        overlap = torch.where(i == ground_truth, i, torch.tensor(0, dtype=i.dtype))
        overlap_frames = torchvision.utils.make_grid(overlap, overlap.size(0))    
    
        plt.imsave(output_dir_pred + f"predictions_{batch_item}.jpeg",
                overlap_frames.cpu().permute(1, 2, 0).numpy())
      
    return 

def calc_black_losses_over_time(target): 
    """Calculate MSE for a black prediction against target over time"""
    black_img = torch.full_like(target[:, 0, :, :, :], 0)
    target = torch.transpose(target, 1, 0) # put Time as 1st dim

    mse = nn.MSELoss(reduction = 'mean') # pixel-wise MSE 
    mse_loss_black = []

    for i in target: 
        mse_loss_black.append(mse(i, black_img).item())
        
    return mse_loss_black

def calc_last_seen_losses_over_time(x, target): 
    """ Calculates MSE over time if we use last seen frame for all
    predicted frames.  

    This is the last frame that the model sees (i.e. final frame of input).
    """
    mse = nn.MSELoss(reduction = 'mean') # pixel-wise MSE 
    mse_last_seen = []
    
    last_seen_frame = x[:,-1,:,:,:]
    target = torch.transpose(target, 1, 0)

    for i in target: 
        mse_last_seen.append(mse(i, last_seen_frame).item())

    return mse_last_seen

def calc_model_losses_over_time(x, target, pred_len, args): 
    """Calculate MSE for each predicted frame over time"""
    kvae = load_kvae(args)

    mse = nn.MSELoss(reduction = 'mean') # pixel-wise MSE 
    mse_loss = []
    
    if args.model == "Hier_KVAE": 
        x_predicted = kvae.predict(x, pred_len)
    else: 
        x_predicted, _, _ = kvae.predict(x, pred_len)
    x_predicted = torch.transpose(x_predicted, 1, 0)
    target = torch.transpose(target, 1, 0)

    for i,j in zip(x_predicted, target): 
        mse_loss.append(mse(i, j).item())
        
    return mse_loss

class Ex1_Args:
    subdirectory = "experiment_hier"
    dataset = "BouncingBall_50"
    model = 'Hier_KVAE' # or KVAE 
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

class Ex2_Args:
    subdirectory = "experiment_hier_2" # not complete yet 
    dataset = "BouncingBall_50"
    model = 'Hier_KVAE' # or KVAE 
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    levels = 2
    factor = 1
    state_dict_path = "saves/BouncingBall_50/kvae_hier/v3/levels=2/factor=1/kvae_state_dict_scale_89.pth"

class Ex3_Args:
    subdirectory = "experiment_hier_3" 
    dataset = "BouncingBall_50"
    model = 'Hier_KVAE' # or KVAE 
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    levels = 2
    factor = 2
    state_dict_path = "saves/BouncingBall_50/kvae_hier/v3/levels=2/factor=2/kvae_state_dict_scale_89.pth"

class Ex4_Args:
    subdirectory = "experiment_hier_4" 
    dataset = "BouncingBall_50"
    model = 'Hier_KVAE' # or KVAE 
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

class Ex5_Args:
    subdirectory = "experiment_hier_5" 
    dataset = "BouncingBall_50"
    model = 'Hier_KVAE' # or KVAE 
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    levels = 3
    factor = 2
    state_dict_path = "saves/BouncingBall_50/kvae_hier/v1/levels=3/factor=2/kvae_state_dict_scale_89.pth"

class KVAE_Args:
    subdirectory = "experiment_kvae"
    dataset = "BouncingBall_50"
    model = 'KVAE' # or KVAE 
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/BouncingBall_50/kvae/v1/attempt2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_80.pth"

class KVAE_Mod_Args:
    subdirectory = "experiment_kvae_mod"
    dataset = "BouncingBall_50"
    model = 'KVAE_mod' # or KVAE 
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/BouncingBall_50/kvae_mod/v1/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_89.pth"

class Modified_ELBO_50_Bonus_Args:
    subdirectory = "experiment_mod_50_bonus"
    dataset = "BouncingBall_50"
    model = 'KVAE'
    alpha = "rnn"
    lstm_layers = 3
    x_dim = 1
    a_dim = 2
    z_dim = 5
    K = 7
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/BouncingBall_50/kvae_mod/bonus/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_99.pth"

if __name__ == "__main__": 
    args1 = Ex1_Args
    args2 = Ex2_Args  
    args3 = Ex3_Args 
    args4 = Ex4_Args
    args5 = Ex5_Args

    args_bonus_mod = Modified_ELBO_50_Bonus_Args # Modified Prior with additional parameters

    args_kvae = KVAE_Args
    args_kvae_mod = KVAE_Mod_Args
    
    data, target = load_dataset("BouncingBall_50", batch_size = 32)
    # data, target = load_dataset("BouncingBall_20", batch_size = 32)

    ### Plot predictions for Bouncing Ball 50 
    # plot_predictions(data, target, 50, args_bb20)
    # plot_predictions_diff_colours(data, target, 50, args1)
    # plot_predictions_overlap(data, target, 50, args_bb20)

    def plot_mse_bb50(): 
        ### MSE over time 
        mse_kvae_hier = calc_model_losses_over_time(data, target, 50, args1)
        mse_kvae_hier2  = calc_model_losses_over_time(data, target, 50, args2)
        mse_kvae_hier3  = calc_model_losses_over_time(data, target, 50, args3)
        mse_kvae_hier4  = calc_model_losses_over_time(data, target, 50, args4)
        mse_kvae_hier5  = calc_model_losses_over_time(data, target, 50, args5)
   
        mse_kvae = calc_model_losses_over_time(data, target, 50, args_kvae)
        mse_kvae_mod = calc_model_losses_over_time(data, target, 50, args_kvae_mod)

        mse_kvae_bonus_mod = calc_model_losses_over_time(data, target, 50, args_bonus_mod)
           
        mse_black = calc_black_losses_over_time(target)
        mse_last_seen = calc_last_seen_losses_over_time(data, target)

        ### Plotting 
        plt.plot(mse_kvae_hier, label="1 level")
        plt.plot(mse_kvae_hier2, label="2 levels, 1 factor")
        plt.plot(mse_kvae_hier3, label="2 levels, 2 factor")
        plt.plot(mse_kvae_hier4, label="3 levels, 1 factor")
        plt.plot(mse_kvae_hier5, label="3 levels, 2 factor")

        plt.plot(mse_kvae, label="KVAE (Standard)")
        # plt.plot(mse_kvae_mod, label = "KVAE(Mod)")
        plt.plot(mse_kvae_bonus_mod, label = "KVAE (Bonus)")
        
        plt.plot(mse_black, label="Black")
        plt.plot(mse_last_seen, label = "Last Seen Frame")

        plt.title("MSE between ground truth and predicted frame over time")
        plt.ylabel('MSE')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(mse_black), 5))
        plt.legend(loc="upper left")

        output_dir = f"plots/BouncingBall_50/KVAE_Hier/"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"KVAE_loss_over_time3.jpeg")
        plt.close('all')

    def plot_mse_bb20(): 
        ### MSE over time 
        mse_kvae_1 = calc_model_losses_over_time(data, target, 50, args_bb20)
        mse_kvae_mod = calc_model_losses_over_time(data, target, 50, args_bb20_mod)

        mse_black = calc_black_losses_over_time(target)
        mse_last_seen = calc_last_seen_losses_over_time(data, target)

        ### Plotting 
        plt.plot(mse_kvae_1, label="KVAE")
        plt.plot(mse_kvae_mod, label="KVAE (Modified Prior)")
        plt.plot(mse_black, label="Black")
        plt.plot(mse_last_seen, label = "Last Seen Frame")

        plt.title("MSE between ground truth and predicted frame over time")
        plt.ylabel('MSE')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(mse_black), 5))
        plt.legend(loc="upper left")

        output_dir = f"plots/BouncingBall_20/KVAE/"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"KVAE_loss_over_time.jpeg")
        plt.close('all')

    plot_mse_bb50()
    # plot_mse_bb20()




