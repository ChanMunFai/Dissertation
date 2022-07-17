### Update inference script to plot visualisations of different KVAE models at once 

import os 
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import torchvision

from kvae.modules import KvaeEncoder, Decoder64, DecoderSimple 
from kvae.elbo_loss import ELBO
from kvae.model_kvae import KalmanVAE
from data.MovingMNIST import MovingMNIST
from dataset.bouncing_ball.bouncing_data import BouncingBallDataLoader

def load_kvae(Args): 
    args = Args()
    kvae = KalmanVAE(args = args).to(args.device)
    state_dict = torch.load(args.state_dict_path, map_location = args.device)
    kvae.load_state_dict(state_dict)

    return kvae 

def call_args(Args): 
    args = Args()
    return args

def load_dataset(dataset, batch_size): 
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

    x_predicted, _, _ = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/KVAE/{args.subdirectory}/predictions/"
        output_dir_gt = f"results/{args.dataset}/KVAE/{args.subdirectory}/ground_truth/"
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

    x_predicted, _, _ = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/KVAE/{args.subdirectory}/Coloured_Predictions/"
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

    x_predicted, _, _ = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/KVAE/{args.subdirectory}/Overlapped_Predictions/"
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
    
    x_predicted, _, _ = kvae.predict(x, pred_len)
    x_predicted = torch.transpose(x_predicted, 1, 0)
    target = torch.transpose(target, 1, 0)

    for i,j in zip(x_predicted, target): 
        mse_loss.append(mse(i, j).item())
        
    return mse_loss

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
    state_dict_path = "saves/BouncingBall_50/kvae/v1/attempt2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_80.pth"

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
    state_dict_path = "saves/BouncingBall_50/kvae/v2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_80.pth"

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

class Ex20_Args:
    subdirectory = "experiment_bb20"
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
    state_dict_path = "saves/BouncingBall_20/kvae/v1/attempt2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_60.pth"

class Bonus_Args:
    subdirectory = "experiment_bonus"
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
    state_dict_path = "saves/BouncingBall_50/kvae/v3/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_89.pth"


args1 = Ex1_Args
args2 = Ex2_Args
args3 = Ex3_Args
args_bb20 = Ex20_Args
args_bonus = Bonus_Args

data, target = load_dataset("BouncingBall_50", batch_size = 32)

### Plot predictions 
plot_predictions(data, target, 50, args_bb20)
plot_predictions_diff_colours(data, target, 50, args_bb20)
plot_predictions_overlap(data, target, 50, args_bb20)

def plot_mse_bb50(): 
    ### MSE over time 
    mse_kvae1 = calc_model_losses_over_time(data, target, 50, args1)
    mse_kvae2 = calc_model_losses_over_time(data, target, 50, args2)
    mse_kvae3 = calc_model_losses_over_time(data, target, 50, args3)
    mse_kvae_bb20 = calc_model_losses_over_time(data, target, 50, args_bb20)
    mse_kvae_bonus = calc_model_losses_over_time(data, target, 50, args_bonus)

    mse_black = calc_black_losses_over_time(target)
    mse_last_seen = calc_last_seen_losses_over_time(data, target)

    ### Plotting 
    plt.plot(mse_kvae1, label="KVAE 1 LSTM")
    plt.plot(mse_kvae2, label="KVAE 2 LSTM")
    plt.plot(mse_kvae3, label = "KVAE 3 LSTM")
    plt.plot(mse_kvae_bb20, label="KVAE (20)")
    plt.plot(mse_kvae_bonus, label="KVAE (Bonus)")
    plt.plot(mse_black, label="Black")
    plt.plot(mse_last_seen, label = "Last Seen Frame")

    plt.title("MSE between ground truth and predicted frame over time")
    plt.ylabel('MSE')
    plt.xlabel('Time')
    plt.xticks(np.arange(0, len(mse_black), 5))
    plt.legend(loc="upper left")

    output_dir = f"plots/BouncingBall_50/KVAE/"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + f"KVAE_loss_over_time.jpeg")
    plt.close('all')

# plot_mse_bb50()



