### Inference on long predictions for best model 
# Best Hierachical Model now: 3 hierachical layers with 1 factor 
# However, best model is actually bonus model with modified prior 

# Use BouncingBall 200 dataset where we see the first 50 frames 
# and then can see how well model performs for next 150 frames 
import os 
import argparse 
import matplotlib.pyplot as plt
import torch
import torchvision

from dataset.bouncing_ball.bouncing_data import BouncingBallDataLoader
from hier_kvae.model_hier_kvae import HierKalmanVAE
from hier_kvae.inference_hier_kvae import *

class Ex4_Args:
    subdirectory = "levels=3_factor=1" 
    dataset = "BouncingBall_200"
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

def load_dataset(dataset, batch_size, seen_len = 50): 
    if dataset == "BouncingBall_200": 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/200/train', seen_len = seen_len)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=False)

    data, target = next(iter(train_loader))
    data = (data - data.min()) / (data.max() - data.min())
    data = torch.where(data > 0.5, 1.0, 0.0)
    target = (target - target.min()) / (target.max() - target.min())
    target = torch.where(target > 0.5, 1.0, 0.0)

    return data, target 

if __name__ == "__main__": 
    args = Ex4_Args

    data, target = load_dataset("BouncingBall_200", batch_size = 32, seen_len = 50)
    # print(data.shape, target.shape)
    
    # plot_predictions(data, target, 150, args)
    # plot_predictions_diff_colours(data, target, 150, args)
    # plot_predictions_overlap(data, target, 150, args)

    def plot_mse_bb200(): 
        ### MSE over time 
        mse_kvae_hier = calc_model_losses_over_time(data, target, 150, args)
        # mse_kvae_standard = calc_model_losses_over_time(data, target, 50, args_kvae)
              
        mse_black = calc_black_losses_over_time(target)
        mse_last_seen = calc_last_seen_losses_over_time(data, target)

        ### Plotting 
        plt.plot(mse_kvae_hier, label="3 levels, 1 factor")    
        plt.plot(mse_black, label="Black")
        plt.plot(mse_last_seen, label = "Last Seen Frame")

        plt.title("MSE between ground truth and predicted frame over time")
        plt.ylabel('MSE')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(mse_black), 10))
        plt.legend(loc="upper left")

        output_dir = f"plots/BouncingBall_200/KVAE_Hier/"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"KVAE_loss_over_time.jpeg")
        plt.close('all')

    def plot_mse_fewshot(): 
        data_1, target_1 = load_dataset("BouncingBall_200", batch_size = 32, seen_len = 1)
        data_5, target_5 = load_dataset("BouncingBall_200", batch_size = 32, seen_len = 5)
        data_10, target_10 = load_dataset("BouncingBall_200", batch_size = 32, seen_len = 10)
        data_20, target_20 = load_dataset("BouncingBall_200", batch_size = 32, seen_len = 20)
        data_100, target_100 = load_dataset("BouncingBall_200", batch_size = 32, seen_len = 100)
        data, target = load_dataset("BouncingBall_200", batch_size = 32, seen_len = 50)

        ### MSE over time 
        mse_kvae_1 = calc_model_losses_over_time(data_1, target_1, 150, args)
        mse_kvae_5 = calc_model_losses_over_time(data_5, target_5, 150, args)
        mse_kvae_10 = calc_model_losses_over_time(data_10, target_10, 150, args)
        mse_kvae_20 = calc_model_losses_over_time(data_20, target_20, 150, args)
        mse_kvae_50 = calc_model_losses_over_time(data, target, 150, args)
        mse_kvae_100 = calc_model_losses_over_time(data, target, 100, args)
              
        mse_black = calc_black_losses_over_time(target)
        mse_last_seen = calc_last_seen_losses_over_time(data, target)

        ### Plotting 
        plt.plot(mse_kvae_1, label="Seen Frames - 1") 
        plt.plot(mse_kvae_5, label="Seen Frames - 5") 
        plt.plot(mse_kvae_10, label="Seen Frames - 10")    
        plt.plot(mse_kvae_20, label="Seen Frames - 20")    
        plt.plot(mse_kvae_50, label="Seen Frames - 50")  
        plt.plot(mse_kvae_100, label="Seen Frames - 100")  
        plt.plot(mse_black, label="Black")
        plt.plot(mse_last_seen, label = "Last Seen Frame")

        plt.title("MSE between ground truth and predicted frame over time")
        plt.ylabel('MSE')
        plt.xlabel('Time')
        plt.xticks(np.arange(0, len(mse_black), 10))
        plt.legend(loc="upper left")

        output_dir = f"plots/BouncingBall_200/KVAE_Hier/"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir + f"KVAE_fewshot.jpeg")
        plt.close('all')


    # plot_mse_bb200()
    plot_mse_fewshot()
    



