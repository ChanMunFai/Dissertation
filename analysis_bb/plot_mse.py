# Plot MSE for SV2P, VRNN (not yet trained), KVAE and HKVAE
# Plot MSE for different variants of HKVAE 

# Use BouncingBall_50 dataset but predict for 150 time steps using BB_100 dataset 

# However, SV2P uses a bigger dataset of 64 X 64 

import torch 
import torch.nn as nn
import torchvision 
import os 
import matplotlib.pyplot as plt 

from dataloader.bouncing_ball import BouncingBallDataLoader
from sv2p.cdna import CDNA 
from sv2p.model_sv2p import PosteriorInferenceNet, LatentVariableSampler
from vrnn.model_vrnn import VRNN
from hier_kvae.inference_hier_kvae import calc_black_losses_over_time, calc_last_seen_losses_over_time, calc_model_losses_over_time, args1_1, args2_1, args2_2, args3_1, args3_2, KVAE_Args, load_kvae
from utils import count_parameters

def load_dataset(dataset, batch_size): 
    if dataset == "BouncingBall_50": 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/50/train')
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=False)

    elif dataset == "BouncingBall_200": 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/200/train', seen_len = 50)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=False)


    elif dataset == "BouncingBall_50_64": # use the 64 X 64 version for SV2P 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/bigger_64/50/train')
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=True)

    elif dataset == "BouncingBall_200_64": # use the 64 X 64 version for SV2P 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/bigger_64/200/train', seen_len = 50)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=batch_size, 
                    shuffle=True)
    else: 
        raise NotImplementedError

    data, target = next(iter(train_loader))
    data = (data - data.min()) / (data.max() - data.min())
    data = torch.where(data > 0.5, 1.0, 0.0)
    target = (target - target.min()) / (target.max() - target.min())
    target = torch.where(target > 0.5, 1.0, 0.0)

    return data, target 

def load_sv2p(state_dict_cdna_path, state_dict_posterior_path, device = "cpu"): 
    state_dict_cdna = torch.load(state_dict_cdna_path, map_location = device)
    state_dict_posterior = torch.load(state_dict_posterior_path, map_location = device)

    model =  CDNA(in_channels = 1, cond_channels = 1,n_masks = 10).to(device) 
    model.load_state_dict(state_dict_cdna)

    q_net = PosteriorInferenceNet(tbatch = 50).to(device)
    q_net.load_state_dict(state_dict_posterior) 

    sampler = LatentVariableSampler()
    
    return model, q_net, sampler 

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

def load_vrnn(state_dict_path, args): 
    state_dict = torch.load(state_dict_path, map_location = args.device)
    
    vrnn = VRNN(args.xdim, args.hdim, args.zdim, args.nlayers).to(args.device)
    vrnn.load_state_dict(state_dict)

    return vrnn 

def calc_mse_over_time(ground_truth, predictions): 
    mse = nn.MSELoss(reduction = 'mean') # pixel-wise MSE 
    mse_loss = []
    
    predictions = torch.transpose(predictions, 1, 0) # time first 
    ground_truth = torch.transpose(ground_truth, 1, 0)

    for i,j in zip(predictions, ground_truth): 
        mse_loss.append(mse(i, j).item())
        
    return mse_loss

def plot_sv2p_predictions(predictions, ground_truth, pred_list=None): 
    for batch_item, i in enumerate(predictions):
        output_dir_pred = "results/BouncingBall_50/sv2p/predictions/"
        output_dir_gt = "results/BouncingBall_50/sv2p/ground_truth/"
        output_dir_coloured = "results/BouncingBall_50/sv2p/Coloured_Predictions/"
        os.makedirs(output_dir_pred, exist_ok = True)
        os.makedirs(output_dir_gt, exist_ok = True)
        os.makedirs(output_dir_coloured, exist_ok = True)

        if pred_list is None: 
            pred_list = torch.arange(0, predictions.size(1))
        
        i = i[pred_list]
        predicted_frames = torchvision.utils.make_grid(i,i.size(0))
        gt = ground_truth[batch_item,pred_list,:,:,:]
        gt_frames = torchvision.utils.make_grid(gt,gt.size(0))

        plt.imsave(output_dir_pred + f"predictions_{batch_item}.jpeg",
                predicted_frames.cpu().permute(1, 2, 0).numpy())

        plt.imsave(output_dir_gt + f"ground_truth_{batch_item}.jpeg",
                gt_frames.cpu().permute(1, 2, 0).numpy())

        # Coloured Predictions 
        empty_channel = torch.full_like(i, 0)
        stitched_video = torch.cat((i, empty_channel, gt), 1)
        stitched_frames = torchvision.utils.make_grid(stitched_video, stitched_video.size(0))    
        plt.imsave(output_dir_coloured + f"predictions_{batch_item}.jpeg",
                stitched_frames.cpu().permute(1, 2, 0).numpy())

    return 

class VRNN_args: 
    model = "VRNN"
    dataset = "BouncingBall_50"
    xdim = 32 
    hdim = 50
    zdim = 50
    nlayers = 1
    device = "cuda"
    state_dict_path = "saves/BouncingBall_50/VRNN/v3/beta=1.0/vrnn_state_dict_140.pth"
    
if __name__ == "__main__": 
    seed = 128
    torch.manual_seed(seed)

    # Datasets 
    data, target = load_dataset("BouncingBall_200_64", batch_size = 32)
    target = target[:,0:150] 
    print(data.size(), target.size())

    ## SV2P
    state_dict_posterior_path = "saves/BouncingBall_50/sv2p/stage3/v1/final_beta=0.001/sv2p_posterior_state_dict_29.pth"
    state_dict_cdna_path = "saves/BouncingBall_50/sv2p/stage3/v1/final_beta=0.001/sv2p_cdna_state_dict_29.pth"
    device = "cpu"
    sv2p, q_net, sampler = load_sv2p(state_dict_cdna_path, state_dict_posterior_path)

    ### VRNN 
    vrnn = load_vrnn(VRNN_args.state_dict_path, VRNN_args)
    print(count_parameters(vrnn)) # 401515

    ### Parameter counts 
    kvae_1_1 = load_kvae(args1_1)
    kvae_3_1 = load_kvae(args3_1)
    print(count_parameters(sv2p) + count_parameters(q_net)) # 9444013
    print(count_parameters(kvae_1_1)) # 72232
    
    params3_1 = count_parameters(kvae_3_1) - (1 * 3 * 4 * 4)
    print(params3_1) # 72424

    predictions_sv2p = sv2p_predict(data, target, num_samples = 1)
    predictions_sv2p = torch.clamp(predictions_sv2p, min = 0, max = 1)
    predictions_sv2p = predictions_sv2p[0] # they all look the same anyway

    mse_sv2p = calc_mse_over_time(target, predictions_sv2p)
    # mse_black = calc_black_losses_over_time(target)
    # mse_last_seen = calc_last_seen_losses_over_time(data, target) 

    pred_list = torch.arange(0, 50, 2)
    # plot_sv2p_predictions(predictions_sv2p, target, pred_list)
    # KVAE  

    ### KVAE
    data_small, target_small = load_dataset("BouncingBall_200", batch_size = 32)
    data_small = data_small.to("cuda")
    target_small = target_small.to("cuda")

    predictions_vrnn = vrnn.predict(data_small, target_small) 
    mse_vrnn = calc_mse_over_time(target_small, predictions_vrnn)

    data_small = data_small.detach().cpu()
    target_small = target_small.detach().cpu()

    mse_kvae = calc_model_losses_over_time(data_small, target_small, target.size(1), args1_1)
    mse_kvae_original = calc_model_losses_over_time(data_small, target_small, target.size(1), KVAE_Args)
    mse_black_small = calc_black_losses_over_time(target_small)
    mse_last_seen_small = calc_last_seen_losses_over_time(data_small, target_small) 

    mse_2_1 = calc_model_losses_over_time(data_small, target_small, target.size(1), args2_1)
    mse_2_2 = calc_model_losses_over_time(data_small, target_small, target.size(1), args2_1)
    mse_3_1 = calc_model_losses_over_time(data_small, target_small, target.size(1), args3_1)
    mse_3_2 = calc_model_losses_over_time(data_small, target_small, target.size(1), args3_2)

    ### Plot all models 
    plt.plot(mse_sv2p, label="SV2P", color = "g")
    plt.plot(mse_vrnn, label="VRNN", color = "c")
    plt.plot(mse_kvae, label = "KVAE", color = 'b')
    plt.plot(mse_3_1, label = "Hier KVAE", color = 'y')
    # plt.plot(mse_kvae_original, label = "KVAE (original)")

    plt.plot(mse_black_small, label = "Black", color = "k")
    plt.plot(mse_last_seen_small, label = "Last Seen", color = "r")

    plt.title("Loss over time: BouncingBalls (all models)")

    plt.ylabel('MSE')
    plt.xlabel('Time')
    plt.legend()
    plt.xticks(np.arange(0, 20, 2))

    output_dir = f"plots/BouncingBall_50/"
    os.makedirs(output_dir, exist_ok = True)
    plt.savefig(output_dir + f"MSE_loss_all_models.jpeg")
    plt.close('all')

    ### Plot all hyperparameters of hierachical KVAE and KVAE 
    # plt.plot(mse_kvae, label = "KVAE", linestyle = "dashed", color = 'b')
    # plt.plot(mse_2_1, label = "level=2, factor=1")
    # plt.plot(mse_2_2, label = "level=2, factor=2")
    # plt.plot(mse_3_1, label = "level=3, factor=1", color = "y")
    # plt.plot(mse_3_2, label = "level=3, factor=2")

    # plt.title("Loss over time: BouncingBalls (Hier KVAE)")

    # plt.ylabel('MSE')
    # plt.xlabel('Time')
    # plt.legend()
    # plt.savefig(output_dir + f"MSE_loss_hierkvae.jpeg")
    # plt.close('all')




    



