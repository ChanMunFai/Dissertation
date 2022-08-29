import os 
import argparse 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import torchvision

torch.manual_seed(288) # 285

from hier_kvae.inference_hier_kvae import load_dataset, load_kvae, calc_model_losses_over_time

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
    factor = 2
    device = "cpu"
    scale = 0.3
    state_dict_path = "saves/DancingMNIST_20_v2/kvae_hier/v5/levels=3/factor=1/kvae_state_dict_scale_89.pth"


class args_3_2: # v5 due to updates in training 
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

def remove_upper_layers(args): 
    """ Performs a forward pass during inference when upper layers are removed.  
    """
    with torch.no_grad(): 
        a_sample, a_mu, a_log_var = kvae._encode(data) 
        A, C, D, weights = kvae._interpolate_matrices(a_sample)

        (T, B, _) = a_sample.size()
        obs = a_sample.unsqueeze(-1)

        mu_filt = torch.zeros(T, B, kvae.z_dim, 1, device = kvae.device, dtype = torch.float) 
        sigma_filt = torch.zeros(T, B, kvae.z_dim, kvae.z_dim, device = kvae.device, dtype = torch.float)
        mu_pred = torch.zeros_like(mu_filt, device = kvae.device, dtype = torch.float)
        sigma_pred = torch.zeros_like(sigma_filt, device = kvae.device, dtype = torch.float)

        mu_t = kvae.mu_0.expand(B,-1).unsqueeze(-1).to(obs.device)
        sigma_t = kvae.sigma_0.expand(B,-1,-1).to(obs.device)

        S_tensor = torch.zeros(T, B, kvae.a_dim, kvae.a_dim, device = kvae.device, dtype = torch.float) 

        for t in range(T): 
            mu_pred[t] = mu_t
            sigma_pred[t] = sigma_t

            y_pred = torch.matmul(C[:,t,:,:], mu_t)
            r = obs[t] - y_pred
            S_t = torch.matmul(torch.matmul(C[:,t,:,:], sigma_t), torch.transpose(C[:,t,:,:], 1,2)) 
            S_t += kvae.R.unsqueeze(0)
            S_tensor[t] = S_t

            Kalman_gain = torch.matmul(torch.matmul(sigma_t, torch.transpose(C[:,t,:,:], 1,2)), torch.inverse(S_t))       
            mu_z = mu_t + torch.matmul(Kalman_gain, r)
            
            I_ = torch.eye(kvae.z_dim).to(obs.device) - torch.matmul(Kalman_gain, C[:, t,:,:])
            sigma_z = torch.matmul(torch.matmul(I_, sigma_t), torch.transpose(I_, 1,2)) + torch.matmul(torch.matmul(Kalman_gain, kvae.R.unsqueeze(0)), torch.transpose(Kalman_gain, 1,2))
            mu_filt[t] = mu_z
            sigma_filt[t] = sigma_z

            if t != T-1:  
                mu_t = torch.matmul(A[:,t+1,l,:,:], mu_z)
                sigma_t = torch.matmul(torch.matmul(A[:,t+1,l,:,:], sigma_z), torch.transpose(A[:,t+1,l,:,:], 1,2))
                sigma_t += kvae.Q.unsqueeze(0)

    return (mu_filt, sigma_filt), (mu_pred, sigma_pred), S_tensor

def pred_removed_upper(args): 
    """ Returns predictions of a hierarchical KVAE with upper levels removed. 
    """
    (B, T, C, H, W) = data.size()
    pred_len = target.size(1)

    # Seen data 
    a_sample, *_ = kvae._encode(data) 
    filtered, pred, S_tensor = remove_upper_layers(args)
    mu_z, sigma_z = filtered  
    
    z_dist = MultivariateNormal(mu_z.squeeze(-1), scale_tril=torch.linalg.cholesky(sigma_z))
    z_sample = z_dist.sample()

    z_sequence = torch.zeros((B, pred_len, kvae.z_dim), device = kvae.device)
    a_sequence = torch.zeros((B, pred_len, kvae.a_dim), device = kvae.device)

    # Unseen data 
    a_t = a_sample[:, -1, :].unsqueeze(1) # BS X T X a_dim
    z_t = z_sample[:, -1, :].to(torch.float32) # BS X T X z_dim

    pred_weights = torch.zeros((B, pred_len, kvae.K), device = kvae.device)

    for t in range(pred_len): 
        if kvae.alpha == "rnn": 
            hidden_state, cell_state = kvae.state_dyn_net
            dyn_emb, kvae.state_dyn_net = kvae.parameter_net(a_t, (hidden_state, cell_state))
            dyn_emb = kvae.alpha_out(dyn_emb)

        elif kvae.alpha == "mlp": 
            dyn_emb = kvae.parameter_net(a_t.reshape(B, -1))

        weights = dyn_emb.softmax(-1).squeeze(1)
        pred_weights[:,t] = weights

        C_t = torch.matmul(weights, kvae.C.reshape(kvae.K, -1)).reshape(B, kvae.a_dim, kvae.z_dim) # BS X z_dim x z_dim 
        A_t = torch.matmul(weights, kvae.A[:,0].reshape(kvae.K, -1)).reshape(B, kvae.z_dim, kvae.z_dim) 
        z_t = torch.matmul(A_t, z_t.unsqueeze(-1)).squeeze(-1) # BS X z_dim 
        a_t = torch.matmul(C_t, z_t.unsqueeze(-1)).squeeze(-1)
        a_t = a_t.unsqueeze(1)
        
        # print(z_sequence[:,t,:].shape, z_t.shape)
        z_sequence[:,t,:] = z_t
        a_sequence[:,t,:] = a_t.squeeze(1)

    return mu_z, z_sequence, a_sequence

def plot_z0(args): 
    """ Plot lowest level of latent variables with and without upper layers removed. 
    """
    output_dir = f"analysis_dmnist/{args.subdirectory}/remove_upper_layers/"
    os.makedirs(output_dir, exist_ok=True)

    mu_z, z_sequence, a_sequence = pred_removed_upper(args)
    mu_z = mu_z[0].squeeze(-1).detach().numpy()
    z_sequence = z_sequence[0].detach().numpy()

    means = np.concatenate((mu_z, z_sequence), axis = 0)

    ### Normal - without upper layers removed 
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

        standard_mean = np.concatenate((hier_mean, j_seq), axis = 1)

    for i in range(means.shape[-1]): 
        plt.plot(means[:,i], label = f"Upper Levels Removed")
        plt.plot(standard_mean[0,:,i], label = "Standard")

        plt.axvline(x = data.shape[1], color = 'black')
        # plt.legend(loc="upper left")
        plt.title(f"z0: Dim={i}")
        plt.savefig(output_dir + f"z0={i}.jpeg")
        plt.close('all')

    return 

def plot_mse_layers_removed(args): 
    output_dir = f"analysis/{args.subdirectory}/mse_remove_upper_layers/"
    os.makedirs(output_dir, exist_ok=True)

    mse_standard = calc_model_losses_over_time(data, target, 20, args)
    mse_removed = calc_losses_removed_layers(data, target, args)

    plt.plot(mse_standard, label = args.subdirectory)
    plt.plot(mse_removed, label = "Removed upper levels")
    plt.legend(loc="upper left")

    plt.savefig(output_dir + f"mse_over_time_32example.jpeg")
    plt.close('all')

def calc_losses_removed_layers(data, target, args): 
    B,pred_len,C,H,W = target.size()

    mse = nn.MSELoss(reduction = 'mean') # pixel-wise MSE 
    mse_loss = []
    
    mu_z, z_sequence, a_sequence = pred_removed_upper(args)
    x_predicted = kvae._decode(a_sequence).reshape(B,pred_len,C,H,W)
    
    x_predicted = torch.transpose(x_predicted, 1, 0)
    target = torch.transpose(target, 1, 0)

    for i,j in zip(x_predicted, target): 
        mse_loss.append(mse(i, j).item())
        
    return mse_loss

def plot_predictions_removed_layers(data, target, pred_len, args): 
    B,pred_len,C,H,W = target.size()
    
    mu_z, z_sequence, a_sequence = pred_removed_upper(args)
    x_predicted = kvae._decode(a_sequence).reshape(B,pred_len,C,H,W)
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"analysis_dmnist/{args.subdirectory}/predictions_removed_upper/"
        if not os.path.exists(output_dir_pred):
            os.makedirs(output_dir_pred)

        predicted_frames = torchvision.utils.make_grid(i,i.size(0))

        plt.imsave(output_dir_pred + f"predictions_{batch_item}.jpeg",
                predicted_frames.cpu().permute(1, 2, 0).numpy())

        ground_truth = target[batch_item,:,:,:,:]
        empty_channel = torch.full_like(i, 0)
        stitched_video = torch.cat((i, empty_channel, ground_truth), 1)
        stitched_frames = torchvision.utils.make_grid(stitched_video, stitched_video.size(0))    
    
        plt.imsave(output_dir_pred + f"coloured_predictions_{batch_item}.jpeg",
                stitched_frames.cpu().permute(1, 2, 0).numpy())

    return 

def plot_predictions(data, target, pred_len, args):
    """ Plot predictions where ground truth and predictions are plotted 
    in different images (and directories)"""

    x_predicted = kvae.predict(data, pred_len)
    print("Size of Predictions:", x_predicted.size())
    
    for batch_item, i in enumerate(x_predicted):
        output_dir_pred = f"analysis_dmnist/{args.subdirectory}/predictions/"
        output_dir_gt = f"analysis_dmnist/{args.subdirectory}/ground_truth/"
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

        empty_channel = torch.full_like(i, 0)
        stitched_video = torch.cat((i, empty_channel, ground_truth), 1)
        stitched_frames = torchvision.utils.make_grid(stitched_video, stitched_video.size(0))    
    
        plt.imsave(output_dir_pred + f"coloured_predictions_{batch_item}.jpeg",
                stitched_frames.cpu().permute(1, 2, 0).numpy())

    return 

if __name__ == "__main__": 
    data, target = load_dataset("DancingMNIST_20_v2", batch_size = 1)
    kvae = load_kvae(args_3_2)
    # filtered, pred, S_tensor = remove_upper_layers(args_3_2)
    # filtered_mean = filtered[0]
    # print(filtered_mean.shape)

    # plot_z0(args_3_1)
    # pred_removed_upper(args_3_1)

    # plot_mse_layers_removed(args_3_2)

    plot_predictions(data, target, 20, args_3_2)
    plot_predictions_removed_layers(data, target, 20, args_3_2)
    
    # plot_predictions_removed_layers(data, target, 20, args_3_2)

