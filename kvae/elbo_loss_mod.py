### Modified to use new implementation of ELBO loss 

import sys
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ELBO():
    def __init__(self, x, x_mu, x_hat, a_sample, a_mu, a_log_var, mu_z_pred, \
            S_tensor, C_t, scale = 0.3):
        """ Object to compute ELBO for Kalman VAE. 

        Here, instead of maximising ELBO, we min -ELBO, whereby 
        
        ELBO = recon_ll - latent_ll + elbo_kf
             = log p(x) - log q(a) + log p(a)

        To min loss, we min: 
        Loss = recon_loss + latent_ll - elbo_kf
             = -log p(x) + log q(a) - log p(a)

        whereby recon_loss = - recon_ll. 

        Arguments: 
            x: Dim [B X T X NC X H X W]
            x_mu: Dim [B X T X NC X H X W]
            x_hat: Dim [B X T X NC X H X W]
            a_sample: Dim [BS X T X a_dim]
            a_mu: Dim [BS X T X a_dim]
            a_log_var: Dim [BS X T X a_dim]
            mu_z_pred: Mean of z_t during Prediction step of Kalman Filtering
                    Dim [BS X T X z_dim X 1]
            S_tensor: Tensor containing S_t, of Dim [BS X T X a_dim X a_dim]
            C_t:Dim [BS X T X a_dim X z_dim]
        """
        self.device = x.device

        self.x = x
        self.x_mu = x_mu 
        self.x_hat = x_hat
        self.a_sample = a_sample 
        self.a_mu = a_mu
        self.a_log_var = a_log_var

        self.mu_z_pred = mu_z_pred.to(self.device) 
        self.S_tensor = S_tensor.to(self.device) 
        self.C_t = C_t.to(self.device) 

        self.scale = scale 
        self.z_dim = self.mu_z_pred.size(2)
        self.a_dim = self.a_mu.size(2)

        # Fixed covariance matrices 
        self.Q = 0.08*torch.eye(self.z_dim).double().to(self.device) 
        self.R = 0.03*torch.eye(self.a_dim).double().to(self.device) 

        # Initialise p(z_1) 
        self.mu_z0 = (torch.zeros(self.z_dim)).double().to(self.device)
        self.sigma_z0 = (20*torch.eye(self.z_dim)).double().to(self.device)

    def compute_loss(self, print_loss = False): 
        """
        Returns: 
            loss: self.scale * recon_loss + latent_ll - elbo_kf
            recon_loss: -log p(x_t|a_t)
                * Can either be MSE, or NLL of a Bernoulli or Gaussian distribution 
            latent_ll: log q(a)
            elbo_kf: log p(a)
        During training, we want loss to go down, recon_loss to go down, latent_ll to go down, 
        elbo_kf to go up. 
        """

        (B, T, *_) = self.x.size()

        recon_loss = self.compute_reconstruction_loss(mode = "bernoulli")
        latent_ll = self.compute_a_marginal_loglikelihood() # log q(a)

        ### LGSSM 
        elbo_kf = self.compute_marginal_likelihood_a()
        # kld = latent_ll - elbo_kf
        loss = self.scale * recon_loss + latent_ll - elbo_kf
        
        if print_loss == True: 
            print("=======> ELBO calculator")
            print("NLL or -log p(x|a) is", recon_loss.item())
            print("log q(a|x) is ", latent_ll.item())
            print("elbo kf is", elbo_kf.item())
            print("loss is", loss)

        # Calculate MSE for tracking 
        mse_loss = self.compute_reconstruction_loss(mode = "mse")

        return loss, recon_loss, latent_ll, elbo_kf, mse_loss  

    def compute_marginal_likelihood_a(self):  

        T, B, z_dim, _ = self.mu_z_pred.size()
        _, _, a_dim, _ = self.S_tensor.size() 

        log_ll_a = 0.0 
        
        for t in range(T): 
            mean = torch.matmul(self.C_t[:, t, :, :], self.mu_z_pred[t]) 
            mean = mean.squeeze(-1) # BS X a_dim X 1 

            var = self.S_tensor[t]
            a_dist = MultivariateNormal(mean, scale_tril=torch.linalg.cholesky(var))
            log_ll = a_dist.log_prob(self.a_sample[:,t]).mean()
            
            log_ll_a += log_ll

        return log_ll_a

    def compute_reconstruction_loss(self, mode = "bernoulli"): 
        """ Compute reconstruction loss of x_hat against x. 
        
        Arguments: 
            mode: 'bernoulli', 'gaussian' or 'mse'

        Returns: 
            recon_loss: Reconstruction Loss summed across all pixels and all time steps, 
                 averaged over batch size. When using 'bernoulli' or 'gaussian', this is 
                 the Negative Log-Likelihood. 
        """
        if mode == "mse": 
            calc_mse_loss = nn.MSELoss(reduction = 'sum').to(self.device) # MSE over all pixels
            mse = calc_mse_loss(self.x, self.x_hat)
            mse = mse / self.x.size(0)
            return mse.to(self.device) 
        
        elif mode == "bernoulli": 
            eps = 1e-5
            prob = torch.clamp(self.x_mu, eps, 1 - eps) # prob = x_mu 
            ll = self.x * torch.log(prob) + (1 - self.x) * torch.log(1-prob)
            ll = ll.mean(dim = 0).sum() 
            return - ll 

        elif mode == "gaussian": 
            x_var = torch.full_like(self.x_mu, 0.01)
            x_dist = MultivariateNormal(self.x_mu, torch.diag_embed(x_var))
            ll = x_dist.log_prob(x).mean(dim = 0).sum() # verify this 
            return - ll
            

    def compute_a_marginal_loglikelihood(self):
        """ Compute q(a). 

        We define the distribution q(.) given its mean and variance. We then 
        find its pdf given a_sample. 

        Arguments: 
            a_sample: Dim [BS X Time X a_dim ]
            a_mu: Dim [BS X Time X a_dim ]
            a_log_var: Dim [BS X Time X a_dim ]

        Returns: 
            latent_ll: q(a_sample)
        """
        a_var = torch.exp(self.a_log_var)
        a_var = torch.clamp(a_var, min = 1e-8) # force values to be above 1e-8
    
        q_a = MultivariateNormal(self.a_mu, torch.diag_embed(a_var)) # BS X T X a_dim
        
        # pdf of a given q_a 
        latent_ll = q_a.log_prob(self.a_sample).mean(dim=0).sum().to(self.device) # summed across all time steps, averaged in batch 
        
        return latent_ll


   
   
