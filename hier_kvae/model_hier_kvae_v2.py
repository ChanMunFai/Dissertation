### Hierachical KVAE v2 
# Attempt to use more efficient implementation 

import argparse 
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli
import matplotlib.pyplot as plt 

from kvae.modules import KvaeEncoder, Decoder64, DecoderSimple, MLP, CNNFastEncoder  
from kvae.elbo_loss_mod import ELBO
from hier_kvae.model_hier_kvae import HierKalmanVAE

class HierKalmanVAE_V2(nn.Module):
    """ More efficient implementation of HierKalmanVAE. 

    Use less cloning functions. 
    # To do: in V3 (or new function), create fewer matrices 

    Matrix A : map states in a level from previous time step to next 
    Matrix C: map lowest states (z_t^0) to observations a_t 
    Matrix D: map states from a higher level to a lower level 

    """
    def __init__(self, *args, **kwargs):
        super(HierKalmanVAE_V2, self).__init__()
        self.args = kwargs['args']
        self.x_dim = self.args.x_dim
        self.a_dim = self.args.a_dim
        self.z_dim = self.args.z_dim
        self.K = self.args.K
        self.scale = self.args.scale
        self.device = self.args.device 
        self.alpha = self.args.alpha 
        self.lstm_layers = self.args.lstm_layers

        # Hierachical 
        self.levels = self.args.levels
        self.factor = self.args.factor 

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

        # Initialise a_1 (optional)
        self.a1 = nn.Parameter(torch.zeros(self.a_dim, requires_grad=True, device = self.device))
        self.state_dyn_net = None

        # Initialise p(z_1) 
        self.mu_0 = torch.zeros(self.z_dim, device = self.device, dtype = torch.float)
        self.sigma_0 = 20*torch.eye(self.z_dim, device = self.device, dtype = torch.float)

        self.A = nn.Parameter(torch.eye(self.z_dim).unsqueeze(0).repeat(self.K,self.levels,1,1).to(self.device))
        self.C = nn.Parameter(torch.randn(self.K, self.a_dim, self.z_dim).to(self.device)*0.05)

        if self.levels != 0: 
            self.D = nn.Parameter(torch.eye(self.z_dim).unsqueeze(0).repeat(self.K,self.levels - 1,1,1).to(self.device)) 
        else: 
            self.D = None 

        # Covariance matrices - fixed. Noise values obtained from paper. 
        self.Q = 0.08*torch.eye(self.z_dim,  device = self.device, dtype = torch.float)
        self.R = 0.03*torch.eye(self.a_dim,  device = self.device, dtype = torch.float)
        self.O = 0.08*torch.eye(self.z_dim,  device = self.device, dtype = torch.float)

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

    def _interpolate_matrices(self, obs):
        (B, T, _) = obs.size()
        
        a1 = self.a1.reshape(1,1,-1).expand(B,-1,-1)
        joint_obs = torch.cat([a1,obs[:,:-1,:]],dim=1)
    
        if self.alpha == "mlp": 
            dyn_emb = self.parameter_net(joint_obs.reshape(B*T, -1))

        elif self.alpha == "rnn": 
            dyn_emb, self.state_dyn_net = self.parameter_net(joint_obs)
            dyn_emb = self.alpha_out(dyn_emb.reshape(B*T,50))
        
        weights = dyn_emb.softmax(-1)
                
        A_t = torch.matmul(weights, self.A.reshape(self.K,-1)).reshape(B,T,self.levels,self.z_dim,self.z_dim)
        C_t = torch.matmul(weights, self.C.reshape(self.K,-1)).reshape(B,T,self.a_dim,self.z_dim)
        D_t = torch.matmul(weights, self.D.reshape(self.K,-1)).reshape(B,T,self.levels-1,self.z_dim,self.z_dim)
        
        return A_t, C_t, D_t, weights 
  
    def hierachical_filter(self, obs, A, C, D = None): 
        
        A = A.to(obs.device)
        C = C.to(obs.device)
        if D != None:
            D = D.to(obs.device)

        (T, B, _) = obs.size()
        obs = obs.unsqueeze(-1)

        mu_filt = torch.zeros(T, B, self.z_dim, 1, device = obs.device, dtype = torch.float)
        sigma_filt = torch.zeros(T, B, self.z_dim, self.z_dim, device = obs.device, dtype = torch.float)
        mu_pred = torch.zeros_like(mu_filt, device = obs.device, dtype = torch.float)
        sigma_pred = torch.zeros_like(sigma_filt, device = obs.device, dtype = torch.float)

        mu_t = self.mu_0.expand(B,-1).unsqueeze(-1).to(obs.device)
        sigma_t = self.sigma_0.expand(B,-1,-1).to(obs.device)

        S_tensor = torch.zeros(T, B, self.a_dim, self.a_dim,  device = obs.device, dtype = torch.float)
        
        if self.levels > 1: 
            mu_j = mu_t.clone() + 0.01 
            sigma_j = self.sigma_0.expand(B, -1,-1).to(self.device)
            latent_means = torch.zeros(T, B, self.levels-1, self.z_dim, 1, device = obs.device, dtype = torch.float)
            latent_variances = torch.zeros(T, B, self.levels-1, self.z_dim, self.z_dim,  device = obs.device, dtype = torch.float)
           
            latent_means[0,:,-1,:,:] = mu_j
            latent_variances[0,:,-1,:,:] = sigma_j
            
        else: 
            latent_means = None 
            latent_variances = None 

        # print(A.shape) # 1 more layer than D - hence need to use a different indexing
        # print(D.shape)

        # Hierachical latents - does not include lowest level of z_t 
        for l in reversed(range(self.levels - 1)): 
            factor_level = self.factor ** (l + 1) 
            # if 3 levels, l are [1, 0]
            # if 2 levels, l are [0]

            for t in range(T): 
                if l == self.levels - 2 and l!=0:  # highest level 
                    if t !=0 and t%factor_level == 0: 
                        latent_means[t,:,l,:,:] = torch.matmul(A[:,t,l+1,:,:], latent_means[t-factor_level,:,l,:,:]) 
                        # latent_variances[t,:,l,:,:] = torch.matmul(torch.matmul(A[:,t,l+1,:,:], latent_variances[t-factor_level,:,l,:,:]), torch.transpose(A[:,t,l+1,:,:], 1,2)) +\
                        #     self.O.unsqueeze(0)

                    elif t%factor_level != 0: # copy over 
                        latent_means[t,:,l,:,:] = latent_means[t-1,:,l,:,:] 
                        latent_variances[t,:,l,:,:] = latent_variances[t-1,:,l,:,:]
                        
                    # print(t, latent_means[t,:,l,:,:], latent_variances[t,:,l,:,:])
                    # print("\n")

                else: # middle levels
                    if t !=0 and t%factor_level == 0: 
                        # latent_means[t,:,l,:,:] = torch.matmul(A[:,t,l+1,:,:], latent_means[t-factor_level,:,l,:,:]) +\
                        #     torch.matmul(D[:,t,l,:,:], latent_means[t,:,l+1,:,:]) 

                        # latent_variances[t,:,l,:,:] = torch.matmul(torch.matmul(A[:,t,l+1,:,:], latent_variances[t-factor_level,:,l,:,:]), torch.transpose(A[:,t,l+1,:,:], 1,2)) + \
                        #     torch.matmul(torch.matmul(D[:,t,l,:,:], latent_variances[t,:,l+1,:,:]), torch.transpose(D[:,t,l,:,:], 1,2))
                        latent_variances[t,:,l,:,:] += self.O.unsqueeze(0)

                    elif t%factor_level != 0: # copy over 
                        latent_means[t,:,l,:,:] = latent_means[t-1,:,l,:,:] 
                        latent_variances[t,:,l,:,:] = latent_variances[t-1,:,l,:,:]
                 
                    # print(l, t, latent_means[t,:,l,:,:], latent_variances[t,:,l,:,:])
                    # print("\n")

        # Standard Kalman Filter 
        for t in range(T): 
            mu_pred[t] = mu_t
            sigma_pred[t] = sigma_t

            y_pred = torch.matmul(C[:,t,:,:], mu_t)
            r = obs[t] - y_pred
            S_t = torch.matmul(torch.matmul(C[:,t,:,:], sigma_t), torch.transpose(C[:,t,:,:], 1,2)) # something wrong here 
            S_t += self.R.unsqueeze(0)
            S_tensor[t] = S_t

            Kalman_gain = torch.matmul(torch.matmul(sigma_t, torch.transpose(C[:,t,:,:], 1,2)), torch.inverse(S_t))       
            mu_z = mu_t + torch.matmul(Kalman_gain, r)
            
            I_ = torch.eye(self.z_dim).to(obs.device) - torch.matmul(Kalman_gain, C[:, t,:,:])
            sigma_z = torch.matmul(torch.matmul(I_, sigma_t), torch.transpose(I_, 1,2)) + torch.matmul(torch.matmul(Kalman_gain, self.R.unsqueeze(0)), torch.transpose(Kalman_gain, 1,2))
            mu_filt[t] = mu_z
            sigma_filt[t] = sigma_z

            if t != T-1:  
                mu_t = torch.matmul(A[:,t+1,0,:,:], mu_z)
                sigma_t = torch.matmul(torch.matmul(A[:,t+1,0,:,:], sigma_z), torch.transpose(A[:,t+1,0,:,:], 1,2))
                sigma_t += self.Q.unsqueeze(0)

                if self.levels > 1: # include hierahical parts 
                    mu_t += torch.matmul(D[:,t+1,0,:,:], latent_means[t,:,1,:,:])
                    sigma_t += torch.matmul(torch.matmul(D[:,t+1,0,:,:], latent_variances[t,:,1,:,:]), torch.transpose(D[:,t+1,0,:,:], 1,2))
                
        return (mu_filt, sigma_filt), (mu_pred, sigma_pred), (latent_means, latent_variances), S_tensor
      
    def hierachical_filter2(self, obs, A, C, D = None): 
        A = A.to(obs.device)
        C = C.to(obs.device)
        if D != None:
            D = D.to(obs.device)

        (T, B, _) = obs.size()
        obs = obs.unsqueeze(-1)

        mu_filt = torch.zeros(T, B, self.z_dim, 1, device = obs.device, dtype = torch.float)
        sigma_filt = torch.zeros(T, B, self.z_dim, self.z_dim, device = obs.device, dtype = torch.float)
        mu_pred = torch.zeros_like(mu_filt, device = obs.device, dtype = torch.float)
        sigma_pred = torch.zeros_like(sigma_filt, device = obs.device, dtype = torch.float)

        mu_t = self.mu_0.expand(B,-1).unsqueeze(-1).to(obs.device)
        sigma_t = self.sigma_0.expand(B,-1,-1).to(obs.device)

        S_tensor = torch.zeros(T, B, self.a_dim, self.a_dim,  device = obs.device, dtype = torch.float)
        
        if self.levels > 1: 
            mu_j = self.mu_0.expand(B,-1).unsqueeze(-1).to(obs.device) + 0.01 
            sigma_j = self.sigma_0.expand(B, -1,-1).to(self.device)
            latent_means = [[mu_j]] # nested list with layer as outer list and time as inner
            latent_variances = [[sigma_j]] 
        
        else: 
            latent_means = None 
            latent_variances = None 

        reversed_idx = 0
        # Hierachical latents - does not include lowest level of z_t 
        for l in reversed(range(self.levels - 1)): 
            factor_level = self.factor ** (l + 1) 
            
            for t in range(T): 
                if l == self.levels - 2 and l!=0:  # highest level 
                    if t !=0 and t%factor_level == 0: 
                        mu_j = torch.matmul(A[:,t,l+1,:,:], latent_means[0][t-factor_level])
                        sigma_j = torch.matmul(torch.matmul(A[:,t,l+1,:,:], sigma_j), torch.transpose(A[:,t,l+1,:,:], 1,2)) +\
                            self.O.unsqueeze(0)

                    if t!= 0: # already recorded
                        latent_means[0].append(mu_j) 
                        latent_variances[0].append(sigma_j)

                else: # middle levels 
                    if t == 0: 
                        mu_j = self.mu_0.expand(B,-1).unsqueeze(-1).to(obs.device) + 0.01 
                        sigma_j = self.sigma_0.expand(B, -1,-1).to(self.device)
                        latent_means.append([mu_j])
                        latent_variances.append([sigma_j])

                    if t !=0 and t%factor_level == 0: 
                        mu_j = torch.matmul(A[:,t,l+1,:,:], latent_means[reversed_idx][t-factor_level]) +\
                            torch.matmul(D[:,t,l,:,:], latent_means[reversed_idx-1][t]) 
                        sigma_j = torch.matmul(torch.matmul(A[:,t,l+1,:,:], latent_variances[reversed_idx][t-factor_level]), torch.transpose(A[:,t,l+1,:,:], 1,2)) +\
                            torch.matmul(torch.matmul(D[:,t,l,:,:], latent_variances[reversed_idx-1][t]), torch.transpose(D[:,t,l,:,:], 1,2)) + self.O.unsqueeze(0)
                            
                    if t != 0: 
                        latent_means[reversed_idx].append(mu_j) 
                        latent_variances[reversed_idx].append(sigma_j)
                    
            reversed_idx +=1
                 
        # Standard Kalman Filter 
        for t in range(T): 
            mu_pred[t] = mu_t
            sigma_pred[t] = sigma_t

            y_pred = torch.matmul(C[:,t,:,:], mu_t)
            r = obs[t] - y_pred
            S_t = torch.matmul(torch.matmul(C[:,t,:,:], sigma_t), torch.transpose(C[:,t,:,:], 1,2)) # something wrong here 
            S_t += self.R.unsqueeze(0)
            S_tensor[t] = S_t

            Kalman_gain = torch.matmul(torch.matmul(sigma_t, torch.transpose(C[:,t,:,:], 1,2)), torch.inverse(S_t))       
            mu_z = mu_t + torch.matmul(Kalman_gain, r)
            
            I_ = torch.eye(self.z_dim).to(obs.device) - torch.matmul(Kalman_gain, C[:, t,:,:])
            sigma_z = torch.matmul(torch.matmul(I_, sigma_t), torch.transpose(I_, 1,2)) + torch.matmul(torch.matmul(Kalman_gain, self.R.unsqueeze(0)), torch.transpose(Kalman_gain, 1,2))
            mu_filt[t] = mu_z
            sigma_filt[t] = sigma_z

            if t != T-1:  
                mu_t = torch.matmul(A[:,t+1,0,:,:], mu_z)
                sigma_t = torch.matmul(torch.matmul(A[:,t+1,0,:,:], sigma_z), torch.transpose(A[:,t+1,0,:,:], 1,2))
                sigma_t += self.Q.unsqueeze(0)

                if self.levels > 1: # include hierahical parts 
                    mu_t += torch.matmul(D[:,t+1,0,:,:], latent_means[-1][t])
                    # sigma_t += torch.matmul(torch.matmul(D[:,t+1,0,:,:], latent_variances[t,:,1,:,:]), torch.transpose(D[:,t+1,0,:,:], 1,2))
                
        return (mu_filt, sigma_filt), (mu_pred, sigma_pred), (latent_means, latent_variances), S_tensor
      
    def predict2(self, input, pred_len, return_weights = False):
        (B, T, C, H, W) = input.size()
       
        with torch.no_grad(): 
            a_sample, *_ = self._encode(input) 
            filtered, pred, hierachical, S_tensor, A_t, C_t, D_t, weights = self._kalman_posterior(a_sample) 

            mu_z, sigma_z = filtered  
            mu_z = torch.transpose(mu_z, 1, 0)
            sigma_z = torch.transpose(sigma_z, 1, 0)
            
            z_dist = MultivariateNormal(mu_z.squeeze(-1), scale_tril=torch.linalg.cholesky(sigma_z))
            z_sample = z_dist.sample()

            if self.levels > 1: 
                j_mean, j_var = hierachical 
                
                # Convert nested list to tensor
                j_mean = torch.stack([torch.stack(l1, dim=0) for l1 in j_mean], dim=0)
                j_var = torch.stack([torch.stack(l1, dim=0) for l1 in j_var], dim=0)
                
                j_mean = torch.permute(j_mean, (2, 1, 0, 3, 4)) # BS X T X layers X z_dim X 1
                j_var = torch.permute(j_var, (2, 1, 0, 3, 4)) # BS X T X layers X z_dim X z_dim
                # lowest index level contains highest level, and vice versa 
                
                j_mean = torch.flip(j_mean, [2]) # verify this is correct 
                j_var = torch.flip(j_var, [2])

                j_sequence = torch.zeros((B, self.levels - 1, pred_len, self.z_dim, 1), device = self.device)
                j_t = j_mean[:, -1,:,:,:].to(torch.float32)
           
            ### Unseen data
            z_sequence = torch.zeros((B, pred_len, self.z_dim), device = self.device)
            a_sequence = torch.zeros((B, pred_len, self.a_dim), device = self.device)
            
            a_t = a_sample[:, -1, :].unsqueeze(1) # BS X T X a_dim
            z_t = z_sample[:, -1, :].to(torch.float32) # BS X T X z_dim
            
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

                C_t = torch.matmul(weights, self.C.reshape(self.K, -1)).reshape(B, self.a_dim, self.z_dim) # BS X z_dim x z_dim 

                for l in reversed(range(self.levels-1)): 
                    factor_level = self.factor ** (l+1)

                    A_t = torch.matmul(weights, self.A[:,l+1].reshape(self.K, -1)).reshape(B, self.z_dim, self.z_dim) # BS X z_dim x z_dim 
                    D_t = torch.matmul(weights, self.D[:,l].reshape(self.K, -1)).reshape(B, self.z_dim, self.z_dim)

                    if l == self.levels - 1 and l!=0: # highest level 
                        if t % factor_level == 0: 
                            j_t[:,l,:,:] = torch.matmul(A_t, j_t[:,l,:,:]) # BS X 4 X 1
                        
                        j_sequence[:,l,t,:,:] = j_t[:,l,:,:] # copy over mean 
                        # print(l)

                    elif l < self.levels -1 and l!=0:
                        if t % factor_level == 0: 
                            j_t[:,l,:,:] = torch.matmul(A_t, j_t[:,l,:,:]) 
                            # print(j_sequence.shape, l+1, t )
                            # + torch.matmul(D_t, j_sequence[:,l+1,t,:,:])
                        
                        j_sequence[:,l,t,:,:] = j_t[:,l,:,:]

                    elif l == 0: 
                        if self.levels == 1: 
                            z_t = torch.matmul(A_t, z_t.unsqueeze(-1)).squeeze(-1) # BS X z_dim 
                        elif self.levels > 1: 
                            z_t = torch.matmul(A_t, z_t.unsqueeze(-1)) + torch.matmul(D_t, j_sequence[:,l+1,t,:,:])
                            z_t = z_t.squeeze(-1)

                        # a_t|z_t
                        a_t = torch.matmul(C_t, z_t.unsqueeze(-1)).squeeze(-1)
                        a_t = a_t.unsqueeze(1)
                        
                        z_sequence[:,t,:] = z_t
                        a_sequence[:,t,:] = a_t.squeeze(1)

            pred_seq = self._decode(a_sequence).reshape(B,pred_len,C,H,W) # BS X pred_len X C X H X W 
    
            return pred_seq

    def predict(self, input, pred_len, return_weights = False):
        (B, T, C, H, W) = input.size()
       
        with torch.no_grad(): 
            a_sample, *_ = self._encode(input) 
            filtered, pred, hierachical, S_tensor, A_t, C_t, D_t, weights = self._kalman_posterior(a_sample) 

            mu_z, sigma_z = filtered  
            mu_z = torch.transpose(mu_z, 1, 0)
            sigma_z = torch.transpose(sigma_z, 1, 0)
            
            z_dist = MultivariateNormal(mu_z.squeeze(-1), scale_tril=torch.linalg.cholesky(sigma_z))
            z_sample = z_dist.sample()

            if self.levels > 1: 
                j_mean, j_var = hierachical 
                j_mean = torch.transpose(j_mean, 1, 0) # BS X T X layers X z_dim X 1
                j_var = torch.transpose(j_var, 1, 0) # BS X T X layers X z_dim X z_dim
                j_sequence = torch.zeros((B, self.levels - 1, pred_len, self.z_dim, 1), device = self.device)
                j_t = j_mean[:, -1,:,:,:].to(torch.float32)
           
            ### Unseen data
            z_sequence = torch.zeros((B, pred_len, self.z_dim), device = self.device)
            a_sequence = torch.zeros((B, pred_len, self.a_dim), device = self.device)
            
            a_t = a_sample[:, -1, :].unsqueeze(1) # BS X T X a_dim
            z_t = z_sample[:, -1, :].to(torch.float32) # BS X T X z_dim
            
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

                C_t = torch.matmul(weights, self.C.reshape(self.K, -1)).reshape(B, self.a_dim, self.z_dim) # BS X z_dim x z_dim 

                for l in reversed(range(self.levels-1)): 
                    factor_level = self.factor ** (l+1)

                    A_t = torch.matmul(weights, self.A[:,l+1].reshape(self.K, -1)).reshape(B, self.z_dim, self.z_dim) # BS X z_dim x z_dim 
                    D_t = torch.matmul(weights, self.D[:,l].reshape(self.K, -1)).reshape(B, self.z_dim, self.z_dim)

                    if l == self.levels - 1 and l!=0: # highest level 
                        if t % factor_level == 0: 
                            j_t[:,l,:,:] = torch.matmul(A_t, j_t[:,l,:,:]) # BS X 4 X 1
                        
                        j_sequence[:,l,t,:,:] = j_t[:,l,:,:] # copy over mean 

                    elif l < self.levels -1 and l!=0:
                        if t % factor_level == 0: 
                            j_t[:,l,:,:] = torch.matmul(A_t, j_t[:,l,:,:]) + torch.matmul(D_t, j_sequence[:,l+1,t,:,:])
                        
                        j_sequence[:,l,t,:,:] = j_t[:,l,:,:]

                    elif l == 0: 
                        if self.levels == 1: 
                            z_t = torch.matmul(A_t, z_t.unsqueeze(-1)).squeeze(-1) # BS X z_dim 
                        elif self.levels > 1: 
                            z_t = torch.matmul(A_t, z_t.unsqueeze(-1)) + torch.matmul(D_t, j_sequence[:,l+1,t,:,:])
                            z_t = z_t.squeeze(-1)

                        # a_t|z_t
                        a_t = torch.matmul(C_t, z_t.unsqueeze(-1)).squeeze(-1)
                        a_t = a_t.unsqueeze(1)
                        
                        z_sequence[:,t,:] = z_t
                        a_sequence[:,t,:] = a_t.squeeze(1)

            pred_seq = self._decode(a_sequence).reshape(B,pred_len,C,H,W) # BS X pred_len X C X H X W 
    
            return pred_seq
    
    def _kalman_posterior(self, obs):
        A, C, D, weights = self._interpolate_matrices(obs)

        filtered, pred, hierachical, S_tensor = self.hierachical_filter2(obs.transpose(1, 0), A, C, D)

        return filtered, pred, hierachical, S_tensor, A, C, D, weights  

    def _decode(self, a):
        """
        Arguments:
            a: Dim [B X T X a_dim]
        
        Returns: 
            x_mu: [B X T X 64 X 64]
        """
        B, T, *_ = a.size()
        x_mu = self.decoder(a)
        return x_mu 

    
        """
        Arguments: 
            obs: Dim [T X B X a_dim]
            A: Dim [B X T X l X z_dim x z_dim]
            D: Dim [B X T X l X z_dim x z_dim]
            C: Dim [B X T X l X a_dim x z_dim]

        Create baseline case for when l = 1, 2, and 3.     
        """

        A = A.to(obs.device)
        D = D.to(obs.device)
        C = C.to(obs.device)

        (T, B, _) = obs.size()
        obs = obs.unsqueeze(-1)

        mu_filt = torch.zeros(T, B, self.z_dim, 1).to(obs.device).double()
        sigma_filt = torch.zeros(T, B, self.z_dim, self.z_dim).to(obs.device).double()
        mu_pred = torch.zeros_like(mu_filt).to(obs.device)
        sigma_pred = torch.zeros_like(sigma_filt).to(obs.device)

        mu_t = self.mu_0.expand(B,-1).unsqueeze(-1).to(obs.device)
        sigma_t = self.sigma_0.expand(B,-1,-1).to(obs.device)

        S_tensor = torch.zeros(T, B, self.a_dim, self.a_dim).double()

        mu_j = mu_t.clone() + 0.01 
        mu_j_original = mu_j.clone() 
        sigma_j = self.sigma_0.expand(B, -1,-1).to(self.device)
        sigma_j_original = sigma_j.clone() 
        latent_means = torch.zeros(T, B, self.levels, self.z_dim, 1).double().to(obs.device)
        latent_variances = torch.zeros(T, B, self.levels, self.z_dim, self.z_dim).double().to(obs.device)

        for l in reversed(range(self.levels)): 
            factor_level = self.factor ** l

            for t in range(T): 
                if l == self.levels - 1 and l!=0: # highest level 
                    if t % factor_level == 0: 
                        if t == 0: 
                            pass 
                        else: 
                            # print(t)
                            mu_j = torch.matmul(A[:,t,l,:,:], mu_j) 
                            sigma_j = torch.matmul(torch.matmul(A[:,t,l,:,:], sigma_j), torch.transpose(A[:,t,l,:,:], 1,2)) 
                            sigma_j += self.O.unsqueeze(0) # verify this 

                    latent_means[t, :, l, :, :] = mu_j.clone() 
                    latent_variances[t, :, l, :, :] = sigma_j.clone() 

                elif l != 0: # middle levels
                    if t % factor_level == 0: 
                        if t == 0: 
                            mu_j = mu_j_original.clone()
                            sigma_j = sigma_j_original.clone()

                        else: 
                            mu_j = torch.matmul(A[:,t,l,:,:], mu_j) 
                            mu_j = mu_j + torch.matmul(D[:,t,l,:,:], latent_means[t,:,l+1,:,:].clone())
                            sigma_j = torch.matmul(torch.matmul(A[:,t,l,:,:], sigma_j), torch.transpose(A[:,t,l,:,:], 1,2)) 
                            sigma_j = sigma_j + torch.matmul(torch.matmul(D[:,t,l,:,:], latent_variances[t,:,l+1,:,:].clone()), torch.transpose(D[:,t,l,:,:], 1,2))
                            sigma_j += self.O.unsqueeze(0) # verify this

                    latent_means[t, :, l, :, :] = mu_j.clone() 
                    latent_variances[t, :, l, :, :] = sigma_j.clone() 

                elif l == 0: 
                    mu_pred[t] = mu_t
                    sigma_pred[t] = sigma_t

                    y_pred = torch.matmul(C[:,t,:,:], mu_t)
                    r = obs[t] - y_pred
                    S_t = torch.matmul(torch.matmul(C[:,t,:,:], sigma_t), torch.transpose(C[:,t,:,:], 1,2)) # something wrong here 
                    S_t += self.R.unsqueeze(0)
                    S_tensor[t] = S_t

                    Kalman_gain = torch.matmul(torch.matmul(sigma_t, torch.transpose(C[:,t,:,:], 1,2)), torch.inverse(S_t))       
                    mu_z = mu_t + torch.matmul(Kalman_gain, r)
                    
                    I_ = torch.eye(self.z_dim).to(obs.device) - torch.matmul(Kalman_gain, C[:, t,:,:])
                    sigma_z = torch.matmul(torch.matmul(I_, sigma_t), torch.transpose(I_, 1,2)) + torch.matmul(torch.matmul(Kalman_gain, self.R.unsqueeze(0)), torch.transpose(Kalman_gain, 1,2))
                    mu_filt[t] = mu_z
                    sigma_filt[t] = sigma_z

                    if t != T-1:  
                        mu_t = torch.matmul(A[:,t+1,l,:,:], mu_z)
                        sigma_t = torch.matmul(torch.matmul(A[:,t+1,l,:,:], sigma_z), torch.transpose(A[:,t+1,l,:,:], 1,2))
                        sigma_t += self.Q.unsqueeze(0)

                        if self.levels > 1: 
                            mu_t += torch.matmul(D[:,t+1,l,:,:], latent_means[t,:,l+1,:,:])
                            sigma_t += torch.matmul(torch.matmul(D[:,t+1,l,:,:], latent_variances[t,:,l+1,:,:]), torch.transpose(D[:,t+1,l,:,:], 1,2))
                        
        return (mu_filt, sigma_filt), (mu_pred, sigma_pred), (latent_means, latent_variances), S_tensor

    def forward(self, x):
        (B,T,C,H,W) = x.size()

        ### Raise warning 
        if self.factor ** self.levels >  T: 
            print("WARNING: temporal scale and/or no. of levels is too large for the data.")

        a_sample, a_mu, a_log_var = self._encode(x) 
        filtered, pred, hierachical, S_tensor, A_t, C_t, D_t, weights = self._kalman_posterior(a_sample)
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
        pred_seq = self.predict(input, input.size(1)) # predict length of input 

        mse_over_time = []

        for t in range(pred_seq.size(1)): 
            mse_over_time.append(calc_mse(pred_seq[:,t], target[:,t]).item())

        avg_mse = np.mean(mse_over_time)

        return avg_mse, mse_over_time

    def reconstruct(self, input): 
        """ Reconstruct x_hat based on input x. 
        """
        (B, T, C, H, W) = input.size()

        with torch.no_grad(): 
            a_sample, _, _ = self._encode(input) 
            x_hat = self._decode(a_sample).reshape(B,T,C,H,W)

        return x_hat 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = "BouncingBall_20", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall]")
    parser.add_argument('--x_dim', default=1, type=int)
    parser.add_argument('--a_dim', default=2, type=int)
    parser.add_argument('--z_dim', default=4, type=int)
    parser.add_argument('--K', default=3, type=int)
    parser.add_argument('--scale', default=0.3, type=float)
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--alpha', default="rnn", type=str, 
                    help = "choose between [mlp, rnn]")
    parser.add_argument('--lstm_layers', default=1, type=int, 
                    help = "Number of LSTM layers. To be used only when alpha is 'rnn'.")
    parser.add_argument('--levels', default=3, type=int, 
                    help = "Number of levels in Linear State Space Model.")
    parser.add_argument('--factor', default=2, type=int, 
                    help = "Temporal Abstraction Factor.")

    args = parser.parse_args()

    kvae = HierKalmanVAE_V2(args = args)
    x_data = torch.rand((args.batch_size, 20, 1, 32, 32))  # BS X T X NC X H X W

    (B,T,C,H,W) = x_data.size()

    # a_sample, a_mu, a_log_var = kvae._encode(x_data) 
    # kvae.A = nn.Parameter(torch.randn(args.K, args.levels, args.z_dim, args.z_dim).to(args.device)*0.05)
    
    # A, C, D, weights = kvae._interpolate_matrices(a_sample)
    # kvae.hierachical_filter(a_sample.transpose(1, 0), A, C, D)

    # with torch.no_grad():
    #     loss, *_ = kvae(x_data)

    kvae.predict2(x_data, 50)


    

    



