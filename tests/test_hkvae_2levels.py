# Test 2 level hier KVAE 
# Hierachical latents copy over if they are not divisible by temporal factor 
# Try for differing numbers of latents 

import argparse 
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import matplotlib.pyplot as plt 

from hier_kvae.model_hier_kvae import HierKalmanVAE

class Args:
    dataset = "BouncingBall_20"
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    scale = 0.3
    device = "cpu"
    alpha = "rnn"
    lstm_layers = 1
    levels = 2
    factor = 2

args = Args()
kvae_hier = HierKalmanVAE(args = args).to(args.device)
kvae_hier.D = nn.Parameter(torch.randn(args.K, args.levels, args.z_dim, args.z_dim).to(args.device))

# Load state dict 
state_dict_path = "saves/BouncingBall_50/kvae_hier/v1/levels=2/factor=2/kvae_state_dict_scale_89.pth"
state_dict = torch.load(state_dict_path, map_location = "cpu")
kvae_hier.load_state_dict(state_dict)
# print(kvae_hier.A.shape) # K X L X 4 X 4
# print(kvae_hier.A)
# Something wrong with this - A for second level (higher) is a unit matrix

x = torch.randn(1, 20, 1, 32, 32) # BS X T X 1 X H X W

def test_forward(): 
    with torch.no_grad(): 
        a_sample, *_ = kvae_hier._encode(x)
        A_hier, C_hier, D_hier, weights_hier = kvae_hier._interpolate_matrices(a_sample)
        filtered_hier, pred_hier, hierachical_hier, S_tensor_hier = kvae_hier.hierachical_filter(a_sample.transpose(1, 0), A_hier, C_hier, D_hier)
        print(A_hier)

        hier_mean = hierachical_hier[0]
        hier_mean = torch.transpose(hier_mean, 1, 0)
        # print(hier_mean.shape)
        # print(hier_mean[0,:,2,:,0])

        hier_var = hierachical_hier[1]
        hier_var = torch.transpose(hier_var, 1, 0)
        # print(hier_var.shape)
        # print(hier_var[0,:,2,:,:])

def test_predict(): 
    with torch.no_grad(): 
        (B, T, C, H, W) = x.shape
        a_sample, *_ = kvae_hier._encode(x) 
        filtered, pred, hierachical, S_tensor, A_t, C_t, D_t, weights = kvae_hier._kalman_posterior(a_sample) 

        # print(A_t.shape, C_t.shape, D_t.shape) 
        # D has its third dimension as the number of levels, but it just requires l -1 dimensions 
        # Because the highest level does not use D 
        # So D[l] should never be used 

        j_mean, j_var = hierachical 
        j_mean = torch.transpose(j_mean, 1, 0) # BS X T X layers X z_dim X 1
        j_var = torch.transpose(j_var, 1, 0) # BS X T X layers X z_dim X z_dim
            
        mu_z, sigma_z = filtered  
        mu_z = torch.transpose(mu_z, 1, 0)
        sigma_z = torch.transpose(sigma_z, 1, 0)

        z_dist = MultivariateNormal(mu_z.squeeze(-1), scale_tril=torch.linalg.cholesky(sigma_z))
        z_sample = z_dist.sample()

        pred_weights = torch.zeros((B, 20, kvae_hier.K))

        z_sequence = torch.zeros((B, 20, args.z_dim))
        a_sequence = torch.zeros((B, 20, args.a_dim))
        a_t = a_sample[:, -1, :].unsqueeze(1)
        z_t = z_sample[:, -1, :].to(torch.float32) 
        j_sequence = torch.zeros((B, args.levels, 20, args.z_dim, 1))
        j_t = j_mean[:, -1,:,:,:].to(torch.float32)

        hidden_state, cell_state = kvae_hier.state_dyn_net
        dyn_emb, kvae_hier.state_dyn_net = kvae_hier.parameter_net(a_t, (hidden_state, cell_state))
        dyn_emb = kvae_hier.alpha_out(dyn_emb)

        weights = dyn_emb.softmax(-1).squeeze(1)
        pred_weights[:,0] = weights

        for t in range(20): 
            for l in reversed(range(args.levels)): 
                factor_level = args.factor ** l

                A_t = torch.matmul(weights, kvae_hier.A[:,l].reshape(kvae_hier.K, -1)).reshape(B, kvae_hier.z_dim, kvae_hier.z_dim) # BS X z_dim x z_dim 
                D_t = torch.matmul(weights, kvae_hier.D[:,l].reshape(kvae_hier.K, -1)).reshape(B, kvae_hier.z_dim, kvae_hier.z_dim)

                if l == args.levels - 1 and l!=0: # highest level 
                    if t % factor_level == 0: 
                        print(A_t)
                        j_t[:,l,:,:] = torch.matmul(A_t, j_t[:,l,:,:]) # BS X 4 X 1
                        # print(j_t[:,l,:,:])

                    j_sequence[:,l,t,:,:] = j_t[:,l,:,:] # copy over mean 

                elif l < args.levels -1 and l!=0:
                    print(2)
                    if t % factor_level == 0: 
                        j_t[:,l,:,:] = torch.matmul(A_t, j_t[:,l,:,:]) + torch.matmul(D_t, j_sequence[:,l+1,t,:,:])
                    
                    j_sequence[:,l,t,:,:] = j_t[:,l,:,:]

                elif l == 0: 
                    if args.levels == 1: 
                        z_t = torch.matmul(A_t, z_t.unsqueeze(-1)).squeeze(-1) # BS X z_dim 
                    elif args.levels > 1: 
                        z_t = torch.matmul(A_t, z_t.unsqueeze(-1)) + torch.matmul(D_t, j_sequence[:,l+1,t,:,:])
                        z_t = z_t.squeeze(-1)

                    # a_t|z_t
                    C_t = torch.matmul(weights, kvae_hier.C.reshape(kvae_hier.K, -1)).reshape(B, kvae_hier.a_dim, kvae_hier.z_dim) # BS X z_dim x z_dim 
                    a_t = torch.matmul(C_t, z_t.unsqueeze(-1))
                    
                    z_sequence[:,t,:] = z_t
                    a_sequence[:,t,:] = a_t.squeeze(-1)

        pred_seq = kvae_hier._decode(a_sequence).reshape(B,20,C,H,W) # BS X pred_len X C X H X W 
    
        # print(pred_seq.shape)
        # print(j_sequence[0,0]) # should be all 0s
        print(j_sequence[0,1])

# test_predict()
test_forward()