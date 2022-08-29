# Modified from: https://github.com/emited/VariationalRecurrentNeuralNetwork

import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

from vrnn.modules import Conv, Deconv, FastEncoder, FastDecoder
from utils import *

# changing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps # numerical logs

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        self.mse_loss = nn.MSELoss(reduction = 'sum') # MSE over all pixels 

        # embedding - embed xt to xt_tilde (dim h_dim)
        if self.h_dim == 1024: 
            self.embed = Conv().to(device)
        else: 
            self.embed = FastEncoder(input_channels = 1, output_dim = self.h_dim).to(device) 
    
        #encoder - encode xt_tilde and h_t-1 into ht
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        # reparameterisation 1 - get mean and variance of zt
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        # decoding - generate xt_hat from h_t-1 and zt_tilde
        self.phi_z = nn.Sequential( # convert zt to zt_tilde (shape: h_dim)
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        if self.h_dim == 1024:
            self.dec = Deconv(h_dim = h_dim).to(device)
        else: 
            self.dec = FastDecoder(input_dim = self.h_dim, output_channels = 1, output_size = 32).to(device)

        #prior - sample zt from h_t-1
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #recurrence - inputs are itself, xt_tilde and h_t-1_tilde
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)

    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        kld_loss = 0
        reconstruction_loss = 0

        h = torch.zeros(self.n_layers, x.size(0), self.h_dim, device=device)

        for t in range(x.size(1)): # sequence length

            xt = x[:,t,:,:,:]
            xt_tilde = self.embed(xt)

            #encoder and reparameterisation
            enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1)) # final layer of h
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

            #decoding
            zt_tilde = self.phi_z(zt) # convert dim from z_dim to h_dim
            input_latent = torch.cat([zt_tilde, h[-1]], 1)
            xt_hat = self.dec(input_latent)

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #recurrence
            _, h = self.rnn(torch.cat([xt_tilde, zt_tilde], 1).unsqueeze(0), h)

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            reconstruction_loss += self.mse_loss(xt_hat, xt)
            # print("MSE Loss at time t:", self.mse_loss(xt_hat, xt))

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)

        reconstruction_loss = reconstruction_loss/x.size(0) # divide by batch size
        # print("MSE loss after 1 forward pass:", reconstruction_loss)

        return kld_loss, reconstruction_loss, \
            (all_enc_mean, all_enc_std)

    def reconstruct(self, x):
        """ Generate reconstructed frames x_t_hat. 
        """
        h = torch.zeros(self.n_layers, x.size(0), self.h_dim, device=device)

        reconstructed_frames = torch.zeros(x.size(1), 1, self.x_dim, self.x_dim, device=device)

        for t in range(x.size(1)):
            xt = x[:,t,:,:,:] # assume x has channel dimension

            xt_tilde = self.embed(xt)

            #encoder and reparameterisation
            enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1)) # final layer of h
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t)

            zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

            #decoding
            zt_tilde = self.phi_z(zt) # convert dim from z_dim to h_dim
            input_latent = torch.cat([zt_tilde, h[-1]], 1)
            xt_hat = self.dec(input_latent)

            #recurrence
            _, h = self.rnn(torch.cat([xt_tilde, zt_tilde], 1).unsqueeze(0), h)

            reconstructed_frames[t] = xt_hat

        return reconstructed_frames

    def predict(self, input, target): 
        total_len = input.size(1) + target.size(1)
        train_len = input.size(1)
        predicted_frames = torch.full_like(target, 0) 
        h = torch.zeros(self.n_layers, input.size(0), self.h_dim, device=device)

        for t in range(total_len):
            if t < input.size(1): # seen data
                xt = input[:,t,:,:,:]
                xt_tilde = self.embed(xt)

                #encoder and reparameterisation
                enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

                #decoding
                zt_tilde = self.phi_z(zt)
                input_latent = torch.cat([zt_tilde, h[-1]], 1)
                xt_hat = self.dec(input_latent)

            else: 
                xt = xt_hat # use predicted xt instead of actual xt
                xt_tilde = self.embed(xt)

                #encoder and reparameterisation
                enc_t = self.enc(torch.cat([xt_tilde, h[-1]], 1))
                enc_mean_t = self.enc_mean(enc_t)
                enc_std_t = self.enc_std(enc_t)

                zt = self._reparameterized_sample(enc_mean_t, enc_std_t)

                #decoding
                zt_tilde = self.phi_z(zt)
                input_latent = torch.cat([zt_tilde, h[-1]], 1)
                xt_hat = self.dec(input_latent)

                predicted_frames[:,t-train_len] = xt_hat

            #recurrence
            _, h = self.rnn(torch.cat([xt_tilde, zt_tilde], 1).unsqueeze(0), h)

        return predicted_frames


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD
        
        https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        """

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return	0.5 * torch.sum(kld_element)

    # Unused functions 

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + torch.log(2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))

if __name__ == "__main__": 
    vrnn = VRNN(x_dim = 32, h_dim = 10, z_dim = 10, n_layers = 1).to(device)
    print("Number of parameters in my Implementation of VRNN", count_parameters(vrnn))

    data = torch.randn(5, 20, 1, 32, 32).to(device)
    vrnn.forward(data)

    

