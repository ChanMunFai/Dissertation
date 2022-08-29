### Test time differences between hierachical KVAE and standard KVAE (modified prior)

import time 
from hier_kvae.model_hier_kvae import HierKalmanVAE
from kvae.model_kvae_mod import KalmanVAEMod
from hier_kvae.inference_hier_kvae import load_dataset
from kvae.elbo_loss_mod import ELBO

import torch

class ArgsHier:
    dataset = "BouncingBall_50"
    model = 'Hier_KVAE' # or KVAE 
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cuda"
    scale = 0.3
    levels = 1
    factor = 1

class ArgsKVAE:
    dataset = "BouncingBall_50"
    model = 'KVAE_Mod' # or KVAE 
    alpha = "rnn"
    lstm_layers = 1
    x_dim = 1
    a_dim = 2
    z_dim = 4
    K = 3
    device = "cuda"
    scale = 0.3

if __name__ == "__main__":  
    args_hier = ArgsHier()
    args_kvae = ArgsKVAE()
    hier_kvae = HierKalmanVAE(args = args_hier).to(args_hier.device)
    kvae = KalmanVAEMod(args = args_kvae).to(args_kvae.device)

    # Load dataset 
    data, target = load_dataset("BouncingBall_50", batch_size = 32)
    B,T,C,H,W = data.size()
    data = data.to(args_hier.device)
    
    ### Time for hier_kvae 
    def time_hier(): 
        print("====> Hierachical KVAE")
        encode_time = time.time()
        a_sample, a_mu, a_log_var = hier_kvae._encode(data) 
        encode_time = time.time() - encode_time
        print("Encode Time:", encode_time)

        posterior_time = time.time()
        filtered, pred, hierachical, S_tensor, A_t, C_t, D_t, weights = hier_kvae._kalman_posterior(a_sample)
        posterior_time = time.time() - posterior_time
        print("Posterior Time:", posterior_time)

        x_hat = hier_kvae._decode(a_sample).reshape(B,T,C,H,W)
        x_mu = x_hat
        
        mu_z_pred = pred[0]
        elbo_calculator = ELBO(data, x_mu, x_hat, 
                            a_sample, a_mu, a_log_var, 
                            mu_z_pred, S_tensor, 
                            C_t, args_hier.scale) 
        
        elbo_time = time.time()
        loss, recon_loss, latent_ll, elbo_kf, mse_loss = elbo_calculator.compute_loss()
        elbo_time = time.time() - elbo_time
        print("ELBO Time:", elbo_time)

        optimiser = torch.optim.Adam(hier_kvae.parameters(), lr=0.007)
        
        backward_time = time.time()
        optimiser.zero_grad()
        loss.backward()
        backward_time = time.time() - backward_time
        print("Backwards Time:", backward_time)

        total_time = encode_time + posterior_time + elbo_time + backward_time
        print("Total Time: ", total_time)

    ### Time for KVAE 
    def time_kvae(): 
        print("====> KVAE (Modified Prior)")
        encode_time = time.time()
        a_sample, a_mu, a_log_var = kvae._encode(data) 
        encode_time = time.time() - encode_time
        print("Encode Time:", encode_time)

        posterior_time = time.time()
        filtered, pred, S_tensor, A_t, C_t, weights = kvae._kalman_posterior(a_sample)
        posterior_time = time.time() - posterior_time
        print("Posterior Time:", posterior_time)

        x_hat = kvae._decode(a_sample).reshape(B,T,C,H,W)
        x_mu = x_hat
        
        mu_z_pred = pred[0]
        elbo_calculator = ELBO(data, x_mu, x_hat, 
                            a_sample, a_mu, a_log_var, 
                            mu_z_pred, S_tensor, 
                            C_t, args_kvae.scale) 
        
        elbo_time = time.time()
        loss, recon_loss, latent_ll, elbo_kf, mse_loss = elbo_calculator.compute_loss()
        elbo_time = time.time() - elbo_time
        print("ELBO Time:", elbo_time)

        optimiser = torch.optim.Adam(kvae.parameters(), lr=0.007)
        
        backward_time = time.time()
        optimiser.zero_grad()
        loss.backward()
        backward_time = time.time() - backward_time
        print("Backwards Time:", backward_time)

        total_time = encode_time + posterior_time + elbo_time + backward_time
        print("Total Time: ", total_time)

    # time_hier()
    # time_kvae()











