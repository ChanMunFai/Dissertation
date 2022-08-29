import os 
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import torchvision

from hier_kvae.inference_hier_kvae import load_dataset, load_kvae

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
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v5/levels=3/factor=1/kvae_state_dict_scale_89.pth"


class args_3_2: 
    subdirectory = "levels=3_factor=2"
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
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v5/levels=3/factor=2/kvae_state_dict_scale_89.pth"

def plot_latents(args): 
    """ Plot latent variables of hier KVAE model for all dimensions (separately) and all levels. 

    Lowest level is plotted separately from upper levels for visual clarity. 
    """
    kvae = load_kvae(args)

    output_dir = f"analysis_dmnist/{args.subdirectory}/"
    os.makedirs(output_dir, exist_ok=True)

    if args.levels == 1: 
        with torch.no_grad(): 
            a_sample, *_ = kvae._encode(data)
            filtered, pred, hierachical, S_tensor, A_t, C_t, D_t, weights = kvae._kalman_posterior(a_sample)

            pred, j_seq = kvae.predict(data, pred_len = 20, return_latents = True)

            filtered_mean = filtered[0].squeeze().detach().numpy()
            j_mean = j_seq.squeeze().detach().numpy()

            means = np.concatenate((filtered_mean, j_mean), axis = 0)

            for idx in range(means.shape[-1]): 
                pred_mean = means[:,idx]
                plt.plot(pred_mean, label = f"z0: dim = {idx}")

                plt.axvline(x = data.shape[1], color = 'black')
                plt.legend(loc="upper left")
                plt.savefig(output_dir + f"z0={idx}.jpeg")
                plt.close('all')

        return 
    
    with torch.no_grad(): 
        a_sample, a_mu, a_log_var = kvae._encode(data) 
        filtered, pred, hierachical, S_tensor, A_t, C_t, D_t, weights = kvae._kalman_posterior(a_sample)
        hier_mean = hierachical[0].squeeze()
        hier_mean = hier_mean.detach().numpy()
        
        hier_mean = np.transpose(hier_mean, (1, 0, 2))
        filtered_mean = filtered[0].squeeze().detach().numpy()
        hier_mean[0] = filtered_mean
        
        pred, j_seq = kvae.predict(data, pred_len = 20, return_latents = True)
        j_seq = j_seq[0,:,:,:,0].detach().numpy() # first item of batch 

        means = np.concatenate((hier_mean, j_seq), axis = 1)
        
        for idx in range(means.shape[-1]): 
            pred_mean = means[:,:,idx]

            for i, level in enumerate(pred_mean): 
                if i == 0: 
                    pass 
                else: 
                    plt.plot(level, label = f"level = {i}")
            
            plt.axvline(x = data.shape[1], color = 'black')
            plt.legend(loc="upper left")
            plt.savefig(output_dir + f"latents_dim={idx}.jpeg")
            plt.close('all')

        # Plot only z_t for l = 0 (lowest level)
        # plot separately due to differences in magnitude 
        for idx in range(means.shape[-1]): 
            pred_mean = means[:,:,idx]

            for i, level in enumerate(pred_mean): 
                if i == 0: 
                    plt.plot(level, label = f"z0: dim = {idx}")
                
            plt.axvline(x = data.shape[1], color = 'black')
            plt.legend(loc="upper left")
            plt.savefig(output_dir + f"z0={idx}.jpeg")
            plt.close('all')

if __name__ == "__main__": 
    data, target = load_dataset("DancingMNIST_20_v2", batch_size = 1)
    plot_latents(args_3_1)

