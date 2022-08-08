# Test that outputs of 1 level hier KVAE is the same as kvae_mod, 
# conditional on input parameters being the same 

import argparse 
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import matplotlib.pyplot as plt 

from kvae.model_kvae_mod import KalmanVAEMod
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
    levels = 1
    factor = 2 # should not affect results for 1 level 

args = Args()
kvae_mod = KalmanVAEMod(args = args).to(args.device)
kvae_hier = HierKalmanVAE(args = args).to(args.device)

# Parameters to standardise - a1, C
kvae_hier.a1 = kvae_mod.a1
kvae_hier.C = kvae_mod.C

x = torch.randn(1, 20, 1, 32, 32) # BS X T X 1 X H X W

def test_interpolate_matrices(): 
    with torch.no_grad(): 
        a_sample, *_ = kvae_mod._encode(x)
        A_mod, C_mod, weights_mod = kvae_mod._interpolate_matrices(a_sample)
        A_hier, C_hier, D_hier, weights_hier = kvae_hier._interpolate_matrices(a_sample)

        # print(torch.allclose(weights_mod, weights_hier)) # False 

        # Standardise weights and interpolated matrices
        filtered_mod, pred_mod, S_tensor_mod = kvae_mod.new_filter_posterior(a_sample.transpose(1, 0), A_hier.squeeze(2), C_hier)
        filtered_hier, pred_hier, hierachical_hier, S_tensor_hier = kvae_hier.hierachical_filter(a_sample.transpose(1, 0), A_hier, C_hier, D_hier)

        filtered_mean_mod = filtered_mod[0]
        filtered_mean_hier = filtered_hier[0]
        filtered_var_mod = filtered_mod[1]
        filtered_var_hier = filtered_hier[1]

        pred_mean_mod = pred_mod[0]
        pred_mean_hier = pred_hier[0]
        pred_var_mod = pred_mod[1]
        pred_var_hier = pred_hier[1]

        # print(filtered_mean_mod.shape, filtered_mean_hier.shape)
        print(torch.allclose(filtered_mean_mod, filtered_mean_hier)) # True 
        print(torch.allclose(filtered_var_mod, filtered_var_hier)) # True
        print(torch.allclose(pred_mean_mod, pred_mean_hier)) # True 
        print(torch.allclose(pred_var_mod, pred_var_hier)) # True

        # print(hierachical_hier) # should be 0 
        # print(hierachical_hier[0].shape) 

        print(torch.allclose(S_tensor_mod.double(), S_tensor_hier.double())) # True 

### Test predict function 
# Predictions should be the same conditional on ... 
# Test with loaded model as well 

def test_prediction(): 
    with torch.no_grad():
        predictions_hier = kvae_hier.predict(x, pred_len = 20)
        predictions_mod, *_ = kvae_mod.predict(x, pred_len = 20)
        # print(torch.allclose(predictions_hier.double(), predictions_mod.double())) #  
        # print(predictions_hier[0,0])
        # print(predictions_mod[0,0])

        ### Do predictions manually 
        (B, T, C, H, W) = x.size()
        a_sample, *_ = kvae_hier._encode(x)
        filtered, pred, hierachical, S_tensor, A_t, C_t, D_t, weights = kvae_hier._kalman_posterior(a_sample) 

        ### KVAE hier 
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
                            j_t[:,l,:,:] = torch.matmul(A_t, j_t[:,l,:,:]) # BS X 4 X 1
                        
                        j_sequence[:,l,t,:,:] = j_t[:,l,:,:] # copy over mean 

                    elif l < args.levels -1 and l!=0:
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
    
        print(pred_seq.shape)

test_prediction()

        
