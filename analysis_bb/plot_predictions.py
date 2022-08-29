# Plot predictions for all model using the same sample (either size = 32 or 64)

from dataloader.bouncing_ball import BouncingBallDataLoader
import torch 
import torch.nn as nn
import torchvision 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 

from analysis_bb.plot_mse import load_kvae, load_sv2p, load_vrnn, VRNN_args
from hier_kvae.inference_hier_kvae import args1_1, args3_1 # arguments for KVAE hier 

def plot_seq(seq, filepath, filename = None, seq_list = None):
    os.makedirs(filepath, exist_ok = True)
    if seq_list is None: 
        seq_list = torch.arange(0, seq.size(1))

    if not torch.is_tensor(seq): 
        seq = torch.tensor(seq)
        if seq.ndim == 4: 
            seq = seq.unsqueeze(0)

    for batch_item, i in enumerate(seq): 
        i = i[seq_list]
        frames = torchvision.utils.make_grid(i,i.size(0))

        if filename is None: 
            plt.imsave(filepath + f"seq_{batch_item}.jpeg",
                frames.cpu().permute(1, 2, 0).numpy())
        else: 
            plt.imsave(filepath + filename + f"{batch_item}.jpeg",
                frames.cpu().permute(1, 2, 0).numpy())

    return 

def plot_2seq(seq1, seq2, filepath, filename = None, seq_list = None):
    os.makedirs(filepath, exist_ok = True)
    if seq_list is None: 
        seq_list = torch.arange(0, seq1.size(1))

    if not torch.is_tensor(seq1): 
        seq1 = torch.tensor(seq1)
        if seq1.ndim == 4: 
            seq1 = seq1.unsqueeze(0)

    for batch_item, i in enumerate(seq1): 
        i = i[seq_list]
        j = seq2[batch_item, seq_list]
        empty_channel = torch.full_like(i, 0)

        stitched_video = torch.cat((i, empty_channel, j), 1)

        frames = torchvision.utils.make_grid(stitched_video,stitched_video.size(0))

        if filename is None: 
            plt.imsave(filepath + f"seq_{batch_item}.jpeg",
                frames.cpu().permute(1, 2, 0).numpy())
        else: 
            plt.imsave(filepath + filename + f"{batch_item}.jpeg",
                frames.cpu().permute(1, 2, 0).numpy())

    return 

def sv2p_predict(data, target, num_samples = 1):
    total_len = data.size(1) + target.size(1)
    batch_size = data.size(0)
    predicted_frames = torch.zeros(num_samples, batch_size, target.size(1), 1, 64, 64, device=data.device)

    for n in range(num_samples): 
        z = sampler.sample_prior((batch_size, 1, 8, 8)).to(data.device)     # Sample latent variables from prior 
 
        hidden = None
        with torch.no_grad():
            for t in range(total_len):
                if t < data.size(1): # seen data
                    x_t = data[:, t, :, :, :]
                    predictions_t, hidden, _, _ = sv2p(inputs = x_t, conditions = z,
                                                    hidden_states=hidden)

                else: 
                    x_t = predictions_t # use predicted x_t instead of actual x_t
                    predictions_t, hidden, _, _ = sv2p(inputs = x_t, conditions = z,
                                                    hidden_states=hidden)
                    predicted_frames[n, :, t-data.size(1)] = predictions_t

    return predicted_frames 



# data = np.load("analysis_bb/BouncingBall_100/samples/input.npz")["arr_0"]
# target = np.load("analysis_bb/BouncingBall_100/samples/target.npz")["arr_0"]
# data_64 = np.load("analysis_bb/BouncingBall_100/samples/input_big.npz")["arr_0"]
# target_64 = np.load("analysis_bb/BouncingBall_100/samples/target_big.npz")["arr_0"]

# data = torch.tensor(data).unsqueeze(0)
# data_64 = torch.tensor(data_64).unsqueeze(0)
# target = torch.tensor(target).unsqueeze(0)
# target_64 = torch.tensor(target_64).unsqueeze(0)

# data = (data - data.min())/(data.max() - data.min())
# data_64 = (data_64 - data_64.min())/(data_64.max() - data_64.min())
# target = (target - target.min())/(target.max() - target.min())
# target_64 = (target_64 - target_64.min())/(target_64.max() - target_64.min())

### Plot VRNN predictions 
# vrnn = load_vrnn(VRNN_args.state_dict_path, VRNN_args)
# predictions_vrnn = vrnn.predict(data.to("cuda").float(), target.to("cuda").float()) 
# plot_2seq(predictions_vrnn, target.to("cuda").float(), 
#     "analysis_bb/BouncingBall_50/predictions/",
#     "vrnn_fifty",  
#     seq_list = torch.arange(0, 50, 2))

# plot_2seq(predictions_vrnn, target.to("cuda").float(), 
#     "analysis_bb/BouncingBall_50/predictions/",
#     "vrnn_onefifty",  
#     seq_list = torch.arange(0, 150, 5))

# Does not work 
# reconstructions_vrnn = vrnn.reconstruct(data.to("cuda").float())
# plot_2seq(reconstructions_vrnn, data.to("cuda").float(), 
#     "analysis_bb/BouncingBall_50/predictions/",
#     "vrnn_recon_fifty",  
#     seq_list = torch.arange(0, 50, 2))

### Plot SV2P predictions 
# state_dict_posterior_path = "saves/BouncingBall_50/sv2p/stage3/v1/final_beta=0.001/sv2p_posterior_state_dict_29.pth"
# state_dict_cdna_path = "saves/BouncingBall_50/sv2p/stage3/v1/final_beta=0.001/sv2p_cdna_state_dict_29.pth"
# device = "cpu"
# sv2p, q_net, sampler = load_sv2p(state_dict_cdna_path, state_dict_posterior_path)

# predictions_sv2p = sv2p_predict(data_64.float(), target_64.float(), num_samples = 1)[0]
# predictions_sv2p = torch.clamp(predictions_sv2p, min = 0, max = 1)
# plot_2seq(predictions_sv2p, target_64.float(), 
#     "analysis_bb/BouncingBall_50/predictions/",
#     "sv2p_onefifty",  
#     seq_list = torch.arange(0, 150, 5))

### Plot KVAE predictions 
# kvae_1_1 = load_kvae(args1_1)
# predictions_kvae = kvae_1_1.predict(data, 150)
# plot_2seq(predictions_kvae, target, 
#     "analysis_bb/BouncingBall_50/predictions/",
#     "kvae_onefifty",  
#     seq_list = torch.arange(0, 150, 5))

### Plot Hier KVAE predictions 
# kvae_3_1 = load_kvae(args3_1)
# predictions_hier_kvae = kvae_3_1.predict(data, 150)
# plot_2seq(predictions_hier_kvae, target, 
#     "analysis_bb/BouncingBall_50/predictions/",
#     "hier_kvae_onefifty",  
#     seq_list = torch.arange(0, 150, 5))

### Plot ground truth 
# plot_seq(target, 
#     "analysis_bb/BouncingBall_50/predictions/",
#     "groundtruth_onefifty",  
#     seq_list = torch.arange(0, 150, 5))