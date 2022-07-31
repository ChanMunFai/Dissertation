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

x = torch.randn(5, 20, 1, 32, 32) # BS X T X 1 X H X W

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
        
