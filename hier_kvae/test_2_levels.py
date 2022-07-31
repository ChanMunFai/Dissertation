# Test that outputs of 2 level hier KVAE 

### Things to test: 
# Hierachical latents copy over if they are not divisible by temporal factor 
# Try for differing numbers of latents 

### Think of a way to make the 2nd level redundant 

# Another test will be to generate some processes

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
    levels = 3
    factor = 2

args = Args()
kvae_hier = HierKalmanVAE(args = args).to(args.device)
kvae_hier.D = nn.Parameter(torch.randn(args.K, args.levels, args.z_dim, args.z_dim).to(args.device))

x = torch.randn(1, 20, 1, 32, 32) # BS X T X 1 X H X W

with torch.no_grad(): 
    a_sample, *_ = kvae_hier._encode(x)
    A_hier, C_hier, D_hier, weights_hier = kvae_hier._interpolate_matrices(a_sample)
    filtered_hier, pred_hier, hierachical_hier, S_tensor_hier = kvae_hier.hierachical_filter(a_sample.transpose(1, 0), A_hier, C_hier, D_hier)

    hier_mean = hierachical_hier[0]
    hier_mean = torch.transpose(hier_mean, 1, 0)
    print(hier_mean.shape)
    # print(hier_mean[0,:,2,:,0])

    hier_var = hierachical_hier[1]
    hier_var = torch.transpose(hier_var, 1, 0)
    print(hier_var.shape)
    # print(hier_var[0,:,2,:,:])