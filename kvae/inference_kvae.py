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

def plot_predictions(x, target, pred_len, plot_len = None):
    x_predicted, _, _ = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/KVAE/{args.subdirectory}/predictions/"
        output_dir_gt = f"results/{args.dataset}/KVAE/{args.subdirectory}/ground_truth/"
        if not os.path.exists(output_dir_pred):
            os.makedirs(output_dir_pred)
        if not os.path.exists(output_dir_gt):
            os.makedirs(output_dir_gt)

        if plot_len == None: 
            plot_len = pred_len

        i = i[:plot_len,:,:,:] 
        predicted_frames = torchvision.utils.make_grid(
                                        i,
                                        i.size(0)
                                        )

        ground_truth = target[batch_item,:plot_len,:,:,:]
        ground_truth_frames = torchvision.utils.make_grid(
                                        ground_truth,
                                        ground_truth.size(0)
                                        )

        stitched_frames = torchvision.utils.make_grid(
                                        [ground_truth_frames, predicted_frames],
                                        1
                                        )

        plt.imsave(
                output_dir_pred + f"predictions_{batch_item}.jpeg",
                predicted_frames.cpu().permute(1, 2, 0).numpy()
                )

        plt.imsave(
                output_dir_gt + f"ground_truth_{batch_item}.jpeg",
                ground_truth_frames.cpu().permute(1, 2, 0).numpy()
                )

def plot_predictions_diff_colours(x, target, pred_len, plot_len = None):
    """ Plot predictions and ground truth in the same image but with different colours.

    Ground truth - blue 
    Predictions - red
    Overlapped regions - blue + red = purple 
    """
    x_predicted, _, _ = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/KVAE/{args.subdirectory}/Coloured_Predictions/"
        if not os.path.exists(output_dir_pred):
            os.makedirs(output_dir_pred)

        if plot_len == None: 
            plot_len = pred_len

        i = i[:plot_len,:,:,:] 
        ground_truth = target[batch_item,:plot_len,:,:,:]
        empty_channel = torch.full_like(i, 0)

        stitched_video = torch.cat((i, empty_channel, ground_truth), 1)
        stitched_frames = torchvision.utils.make_grid(stitched_video, stitched_video.size(0))    
    
        plt.imsave(
                output_dir_pred + f"predictions_{batch_item}.jpeg",
                stitched_frames.cpu().permute(1, 2, 0).numpy()
                )

def plot_predictions_overlap(x, target, pred_len, plot_len = None):
    """ Plot overlaps of predictions and ground truth.
    """
    x_predicted, _, _ = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"results/{args.dataset}/KVAE/{args.subdirectory}/Overlapped_Predictions/"
        if not os.path.exists(output_dir_pred):
            os.makedirs(output_dir_pred)

        if plot_len == None: 
            plot_len = pred_len

        i = i[:plot_len,:,:,:] 
        ground_truth = target[batch_item,:plot_len,:,:,:]
        overlap = torch.where(i == ground_truth, i, torch.tensor(0, dtype=i.dtype))
        overlap_frames = torchvision.utils.make_grid(overlap, overlap.size(0))    
    
        plt.imsave(
                output_dir_pred + f"predictions_{batch_item}.jpeg",
                overlap_frames.cpu().permute(1, 2, 0).numpy()
                )
        


        




def plot_reconstructions(x, plot_len, reconstruct_kalman = True):
    if reconstruct_kalman == True: 
        x_reconstructed = kvae.reconstruct_kalman(x)
    else: 
        x_reconstructed = kvae.reconstruct(x)
    
    for batch_item, i  in enumerate(x_reconstructed):
        if reconstruct_kalman == False: 
            output_dir = f"results/{args.dataset}/KVAE/{args.subdirectory}/reconstructions/"
        else: 
            output_dir = f"results/{args.dataset}/KVAE/{args.subdirectory}/reconstructions_kf/"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        i = i[:plot_len,:,:,:] 

        reconstructed_frames = torchvision.utils.make_grid(
                                        i,
                                        i.size(0)
                                        )

        ground_truth = x[batch_item,:plot_len,:,:,:]
        ground_truth_frames = torchvision.utils.make_grid(
                                        ground_truth,
                                        ground_truth.size(0)
                                        )

        stitched_frames = torchvision.utils.make_grid(
                                        [ground_truth_frames, reconstructed_frames],
                                        1
                                        )

        plt.imsave(
                output_dir + f"reconstructions_{batch_item}.jpeg",
                stitched_frames.cpu().permute(1, 2, 0).numpy()
                )



def calc_losses_over_time(x, target, pred_len): 
    """Calculate MSE for each predicted frame
    over time"""

    mse = nn.MSELoss(reduction = 'mean') # pixel-wise MSE 
    mse_loss = []
    mse_loss_black = []
    mse_loss_last_seen = []

    x_predicted, _, _ = kvae.predict(x, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    x_predicted = torch.transpose(x_predicted, 1, 0)
    target = torch.transpose(target, 1, 0)

    ### Black Image 
    black_img = torch.full_like(x[:, 0, :, :, :], 0) # batch of black image for 1 time step

    ### Last Seen Frame 
    last_seen_frame = x[:,-1,:,:,:]

    for i,j in zip(x_predicted, target): 
        mse_loss.append(mse(i, j).item())
        mse_loss_black.append(mse(i, black_img).item())
        mse_loss_last_seen.append(mse(i, last_seen_frame).item())

    return mse_loss, mse_loss_black, mse_loss_last_seen
    
if __name__ == "__main__": 
    # state_dict_path = "saves/BouncingBall_50/kvae/v2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_80.pth" 
    state_dict_path = "saves/BouncingBall_20/kvae/v1/attempt2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_60.pth"
    # state_dict_path = "saves/BouncingBall_50/kvae/v1/attempt2/scale=0.3/scheduler_step=20/kvae_state_dict_scale=0.3_80.pth"

    parser = argparse.ArgumentParser()
    parser.add_argument('--subdirectory', default="experiment_bb20", type=str)

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

    kvae = KalmanVAE(args = args).to(args.device)
    state_dict = torch.load(state_dict_path, map_location = args.device)
    kvae.load_state_dict(state_dict)

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

    ### Plot predictions 
    # plot_predictions(data, target, pred_len = 50)  
    plot_predictions_diff_colours(data, target, pred_len = 50)
    plot_predictions_overlap(data, target, pred_len = 50)

    ### Calculate MSE for prediction over time (no plotting)
    # mse_kvae, mse_black, mse_last_seen = calc_losses_over_time(data, target, pred_len = 50)
 
    ### Plotting 
    # plt.plot(mse_kvae, label="KVAE")
    # plt.plot(mse_black, label="Black")
    # plt.plot(mse_last_seen, label = "Last Seen Frame")

    # plt.title("MSE between ground truth and predicted frame over time")
    # plt.ylabel('MSE')
    # plt.xlabel('Time')
    # plt.xticks(np.arange(0, len(mse_kvae), 5))
    # plt.legend(loc="upper left")

    # output_dir = f"plots/{args.dataset}/KVAE/"
    # if not os.path.isdir(output_dir):
    #     os.makedirs(output_dir)
    # plt.savefig(output_dir + f"KVAE_loss_over_time.jpeg")
    # plt.close('all')






    
        


        