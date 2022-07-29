"""Modified to use new implmentation of ELBO. 

Calculates p(a) differently. 
"""

import argparse 
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import matplotlib.pyplot as plt 

from kvae.modules import KvaeEncoder, Decoder64, DecoderSimple, MLP, CNNFastEncoder  
from kvae.elbo_loss_mod import ELBO

class KalmanVAEMod(nn.Module):
    def __init__(self, *args, **kwargs):
        super(KalmanVAEMod, self).__init__()
        print("Using KVAE with alternative prior implementation.")

        self.args = kwargs['args']
        self.x_dim = self.args.x_dim
        self.a_dim = self.args.a_dim
        self.z_dim = self.args.z_dim
        self.K = self.args.K
        self.scale = self.args.scale
        self.device = self.args.device 
        self.alpha = self.args.alpha 
        self.lstm_layers = self.args.lstm_layers

        if self.args.dataset == "MovingMNIST": 
            self.encoder = KvaeEncoder(input_channels=1, input_size = 64, a_dim = 2).to(self.device)
            self.decoder = DecoderSimple(input_dim = 2, output_channels = 1, output_size = 64).to(self.device)
        elif self.args.dataset == "BouncingBall_20" or self.args.dataset == "BouncingBall_50": 
            self.encoder = CNNFastEncoder(1, self.a_dim).to(self.device)
            self.decoder = DecoderSimple(input_dim = 2, output_channels = 1, output_size = 32).to(self.device)

        if self.alpha == "mlp": 
            self.parameter_net = MLP(self.a_dim, 50, self.K).to(self.device)
        else:  
            self.parameter_net = nn.LSTM(self.a_dim, 50, self.lstm_layers, batch_first=True).to(self.device) 
            self.alpha_out = nn.Linear(50, self.K).to(self.device)

        # Initialise a_1 
        self.a1 = nn.Parameter(torch.zeros(self.a_dim, requires_grad=True, device = self.device))
        self.state_dyn_net = None

        # Initialise p(z_1) 
        self.mu_0 = (torch.zeros(self.z_dim)).double()
        self.sigma_0 = (20*torch.eye(self.z_dim)).double()

        # A initialised with identity matrices. B initialised from Gaussian 
        self.A = nn.Parameter(torch.eye(self.z_dim).unsqueeze(0).repeat(self.K,1,1).to(self.device))
        self.C = nn.Parameter(torch.randn(self.K, self.a_dim, self.z_dim).to(self.device)*0.05)

        # Covariance matrices - fixed. Noise values obtained from paper. 
        self.Q = 0.08*torch.eye(self.z_dim).double().to(self.device) 
        self.R = 0.03*torch.eye(self.a_dim).double().to(self.device) 

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _encode(self, x):
        """ Encodes observations x into a. 

        Arguments: 
            x: input data of shape [BS X T X NC X H X W]
        
        Returns: 
            a_sample: shape [BS X T X a_dim]
            a_mu: shape [BS X T X a_dim]
            a_log_var: shape [BS X T X a_dim]
        """
        (a_mu, a_log_var, _) = self.encoder(x)
        eps = torch.normal(mean=torch.zeros_like(a_mu)).to(x.device)
        a_std = (a_log_var*0.5).exp()
        a_sample = a_mu + a_std*eps
        a_sample = a_sample.to(x.device)
        
        return a_sample, a_mu, a_log_var

    def _interpolate_matrices(self, obs, learn_a1 = True):
        """ Generate weights to choose and interpolate between K operating modes. 

        Dynamic parameters network generates weights that sums to 1 for each K operating mode. 
        The weights are of dimension [BS * T X K]. For each item in a time step and in a batch, 
        we have K different weights. 

        Weights are multiplied with A and C matrices to produce At and Ct respectively.
        
        A and C matrices are different state transition and emission matrices for K different modes. 
        In contrast, At and Ct are the interpolated (weighted) versions of them. 
        
        Parameters:
            obs: vector a of dimension [BS X T X a_dim]
            learn_a1: bool. If learn_a1 is True, we replace a_1 with a learned embedding. 
                Otherwise, we generate a_1 from encoding x_1. 

        Returns: 
            A_t: interpolated state transition matrix of dimension [BS X T X z_dim X z_dim]
            C_t: interpolated state transition matrix of dimension [BS X T X a_dim X z_dim]

        """
        (B, T, _) = obs.size()
        
        if learn_a1: 
            a1 = self.a1.reshape(1,1,-1).expand(B,-1,-1)
            joint_obs = torch.cat([a1,obs[:,:-1,:]],dim=1)
            # print("Initialised a1 shape", a1.shape) # BS X 1 X a_dim
            # print("joint_obs shape", joint_obs.shape) # BS X T X a_dim

        else: 
            joint_obs = obs
            # print("joint_obs shape", joint_obs.shape) # BS X T X a_dim

        if self.alpha == "mlp": 
            dyn_emb = self.parameter_net(joint_obs.reshape(B*T, -1))

        elif self.alpha == "rnn": 
            dyn_emb, self.state_dyn_net = self.parameter_net(joint_obs)
            dyn_emb = self.alpha_out(dyn_emb.reshape(B*T,50))
        
        weights = dyn_emb.softmax(-1)
        
        # print("Weights shape", weights.shape) # B*T X K 
        # print("A shape", self.A.shape) # K X z_dim X z_dim  
        # print("C shape", self.C.shape) # K X a_dim X z_dim  
        
        A_t = torch.matmul(weights, self.A.reshape(self.K,-1)).reshape(B,T,self.z_dim,self.z_dim).double()
        C_t = torch.matmul(weights, self.C.reshape(self.K,-1)).reshape(B,T,self.a_dim,self.z_dim).double()
        
        return A_t, C_t, weights 
    
    def _kalman_posterior(self, obs):
        
        A, C, weights = self._interpolate_matrices(obs)

        # New code 
        filtered, pred, S_tensor = self.new_filter_posterior(obs.transpose(1, 0), A, C)

        return filtered, pred, S_tensor, A, C, weights  

    def _decode(self, a):
        """
        Arguments:
            a: Dim [B X T X a_dim]
        
        Returns: 
            x_mu: [B X T X 64 X 64]
        """
        B, T, *_ = a.size()
        x_mu = self.decoder(a)
        # print("Decoded x:", x_mu.size())

        return x_mu 
        
    def forward(self, x):
        (B,T,C,H,W) = x.size()

        a_sample, a_mu, a_log_var = self._encode(x) 
        
        filtered, pred, S_tensor, A_t, C_t, weights  = self._kalman_posterior(a_sample)
        mu_z_pred = pred[0]

        x_hat = self._decode(a_sample).reshape(B,T,C,H,W)
        x_mu = x_hat # assume they are the same for now

        averaged_weights = self._average_weights(weights)

        # Calculate variance of weights 
        weights = weights.reshape(B, T, self.K)
        
        var_diff = []
        for item in weights: # each item in a batch 
            diff_item = [] 
            for t in item: 
                diff_item.append(torch.max(t).item() - torch.min(t).item())
                var_diff.append(np.var(diff_item)) # variance across all time steps
        var_diff = np.mean(var_diff)
        
        # ELBO
        elbo_calculator = ELBO(x, x_mu, x_hat, 
                        a_sample, a_mu, a_log_var, 
                        mu_z_pred, S_tensor, 
                        C_t, self.scale)
        loss, recon_loss, latent_ll, elbo_kf, mse_loss = elbo_calculator.compute_loss()

        return loss, recon_loss, latent_ll, elbo_kf, mse_loss, averaged_weights, var_diff 

    def reconstruct(self, input): 
        """ Reconstruct x_hat based on input x. 
        """
        (B, T, C, H, W) = input.size()

        with torch.no_grad(): 
            a_sample, _, _ = self._encode(input) 
            x_hat = self._decode(a_sample).reshape(B,T,C,H,W)

        return x_hat 

    def predict(self, input, pred_len, return_weights = False):
        """ Predicts a sequence of length pred_len given input. 
        """

        # To be tested

        ### Seen data
        (B, T, C, H, W) = input.size()

        with torch.no_grad(): 
            a_sample, *_ = self._encode(input) 
            filtered, pred, S_tensor, A_t, C_t, weights = self._kalman_posterior(a_sample) 
           
            mu_z, sigma_z = filtered  
            mu_z = torch.transpose(mu_z, 1, 0)
            sigma_z = torch.transpose(sigma_z, 1, 0)

            z_dist = MultivariateNormal(mu_z.squeeze(-1), scale_tril=torch.linalg.cholesky(sigma_z))
            z_sample = z_dist.sample()

            ### Unseen data
            z_sequence = torch.zeros((B, pred_len, self.z_dim), device = self.device)
            a_sequence = torch.zeros((B, pred_len, self.a_dim), device = self.device)
            a_t = a_sample[:, -1, :].unsqueeze(1) # BS X T X a_dim
            z_t = z_sample[:, -1, :].unsqueeze(1).to(torch.float32) # BS X T X z_dim

            pred_weights = torch.zeros((B, pred_len, self.K), device = self.device)

            for t in range(pred_len):
                if self.alpha == "rnn": 
                    hidden_state, cell_state = self.state_dyn_net
                    dyn_emb, self.state_dyn_net = self.parameter_net(a_t, (hidden_state, cell_state))
                    dyn_emb = self.alpha_out(dyn_emb)

                elif self.alpha == "mlp": 
                    dyn_emb = self.parameter_net(a_t.reshape(B, -1))

                weights = dyn_emb.softmax(-1).squeeze(1)
                pred_weights[:,t] = weights
                
                A_t = torch.matmul(weights, self.A.reshape(self.K,-1)).reshape(B,-1,self.z_dim,self.z_dim) # only for 1 time step 
                C_t = torch.matmul(weights, self.C.reshape(self.K,-1)).reshape(B,-1,self.a_dim,self.z_dim)

                # Generate z_t+1
                z_t = torch.matmul(A_t, z_t.unsqueeze(-1)).squeeze(-1)
                z_sequence[:,t,:] = z_t.squeeze(1) 

                # Generate a_t 
                a_t = torch.matmul(C_t, z_t.unsqueeze(-1)).squeeze(-1)
                a_sequence[:,t,:] = a_t.squeeze(1)

            pred_seq = self._decode(a_sequence).reshape(B,pred_len,C,H,W)

        if return_weights == True: 
            return pred_seq, a_sequence, z_sequence, pred_weights
        
        return pred_seq, a_sequence, z_sequence

    def _average_weights(self,weights):
        """ Plot weights 
        Args: 
            weights: dim [B*T X K]

        Returns: 
            fig: Matplotlib object  
        """
        averaged_weights = torch.mean(weights, axis = 0)
        averaged_weights = averaged_weights.tolist()
        
        return averaged_weights

    def new_filter_posterior(self, obs, A, C):
        # obs: (T ,B, N)
        # A: (T, D, D)
        # C: (T, N, D)
        A = A.to(self.device)
        C = C.to(self.device)

        (T, B, _) = obs.size()
        mu_filt = torch.zeros(T, B, self.z_dim, 1).to(obs.device).double()
        Sigma_filt = torch.zeros(T, B, self.z_dim, self.z_dim).to(obs.device).double()
        obs = obs.unsqueeze(-1)
        mu_t = self.mu_0.expand(B,-1).unsqueeze(-1).to(self.device)
        Sigma_t = self.sigma_0.expand(B,-1,-1).to(self.device)

        mu_pred = torch.zeros_like(mu_filt).to(self.device)
        Sigma_pred = torch.zeros_like(Sigma_filt).to(self.device)

        S_tensor = torch.zeros(T, B, self.a_dim, self.a_dim)

        for t in range(T):
            
            # mu/sigma: t | t-1
            mu_pred[t] = mu_t
            Sigma_pred[t] = Sigma_t

            y_pred = torch.matmul(C[:,t,:,:], mu_t)
            r = obs[t] - y_pred
            S_t = torch.matmul(torch.matmul(C[:,t,:,:], Sigma_t), torch.transpose(C[:,t,:,:], 1,2))
            S_t += self.R.unsqueeze(0)
            S_tensor[t] = S_t 

            Kalman_gain = torch.matmul(torch.matmul(Sigma_t, torch.transpose(C[:,t,:,:], 1,2)), torch.inverse(S_t))       
            # filter: t | t
            mu_z = mu_t + torch.matmul(Kalman_gain, r)
            
            I_ = torch.eye(self.z_dim).to(obs.device) - torch.matmul(Kalman_gain, C[:,t,:,:])
            Sigma_z = torch.matmul(torch.matmul(I_, Sigma_t), torch.transpose(I_, 1,2)) + torch.matmul(torch.matmul(Kalman_gain, self.R.unsqueeze(0)), torch.transpose(Kalman_gain, 1,2))
            mu_filt[t] = mu_z
            Sigma_filt[t] = Sigma_z
            if t != T-1:
                # mu/sigma: t+1 | t for next step
                mu_t = torch.matmul(A[:,t+1,:,:], mu_z)
                Sigma_t = torch.matmul(torch.matmul(A[:,t+1,:,:], Sigma_z), torch.transpose(A[:,t+1,:,:], 1,2))
                Sigma_t += self.Q.unsqueeze(0)

        return (mu_filt, Sigma_filt), (mu_pred, Sigma_pred), S_tensor

    def calc_pred_mse(self, input, target):
        """ Calculate MSE between prediction and ground truth. 

        Arguments: 
            input: Dim [B X T X N X H X W]
            target: Dim [B X T X N X H X W]

        Returns: 
            avg_mse[float]: Pixel-wise MSE between input and target, 
                            averaged acrossed all time frames 
            mse_over_time[list]: MSE for each time step 
        """ 
        calc_mse = nn.MSELoss(reduction = 'mean') # pixel-wise MSE 
        pred_seq, *_ = self.predict(input, input.size(1)) # predict length of input 
        
        mse_over_time = []

        for t in range(pred_seq.size(1)): 
            mse_over_time.append(calc_mse(pred_seq[:,t], target[:,t]).item())

        avg_mse = np.mean(mse_over_time)

        return avg_mse, mse_over_time

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = "BouncingBall_20", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall]")
    parser.add_argument('--x_dim', default=1, type=int)
    parser.add_argument('--a_dim', default=2, type=int)
    parser.add_argument('--z_dim', default=4, type=int)
    parser.add_argument('--K', default=3, type=int)
    parser.add_argument('--scale', default=0.3, type=float)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--alpha', default="rnn", type=str, 
                    help = "choose between [mlp, rnn]")
    parser.add_argument('--lstm_layers', default=1, type=int, 
                    help = "Number of LSTM layers. To be used only when alpha is 'rnn'.")

    args = parser.parse_args()

    kvae = KalmanVAE(args = args)
    x_data = torch.rand((5, 10, 1, 32, 32))  # BS X T X NC X H X W

    # with torch.no_grad(): 
    #     kvae(x_data)

    with torch.no_grad(): 
        B, T, C, H, W = x_data.size()
        a_sample, a_mu, a_log_var = kvae._encode(x_data) 
        # A, C, weights = kvae._interpolate_matrices(a_sample) 
        # filtered, pred, S_tensor = kvae.new_filter_posterior(a_sample.transpose(1, 0), A, C)
        mu_z_pred, S_tensor, A_t, C_t, weights  = kvae._kalman_posterior(a_sample)
        x_hat = kvae._decode(a_sample).reshape(B,T,C,H,W)
        x_mu = x_hat # assume they are the same for now
        elbo_calculator = ELBO(x_data, x_mu, x_hat, 
                        a_sample, a_mu, a_log_var, 
                        mu_z_pred, S_tensor, 
                        A_t, C_t)
        # elbo_calculator.calc_marginal_likelihood_a()
        loss, recon_loss, latent_ll, elbo_kf, mse_loss = elbo_calculator.compute_loss()


        