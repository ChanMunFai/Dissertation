import torch
import torch.distributions
import torch.nn as nn

from dataloader.MovingMNIST import MovingMNISTDataLoader
from sv2p.cdna import CDNA # network for CDNA
from sv2p.model_sv2p import PosteriorInferenceNet, LatentVariableSampler

from utils import kld_gauss, kld_standard_gauss

seed = 128 
torch.manual_seed(seed)
batch_size = 52

train_set = MovingMNISTDataLoader(root='dataset/mnist', train=True, download=True)
train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True)

q_net = PosteriorInferenceNet(tbatch = 10)
sampler = LatentVariableSampler()
stoc_model = CDNA(in_channels = 1, cond_channels = 1,
            n_masks = 10)

def test_KL_divergence(): 
    for data, _ in train_loader:
        data = torch.unsqueeze(data, 2) # Batch Size X Seq Length X Channels X Height X Width
        data = (data - data.min()) / (data.max() - data.min())

        mu, sigma = q_net(data) 

        prior_mean = torch.full_like(mu, 0)
        prior_std = torch.full_like(sigma, 1)
        
        p = torch.distributions.Normal(mu,sigma)
        q = torch.distributions.Normal(prior_mean,prior_std)

        out = torch.distributions.kl_divergence(p, q)
        # print(out)
        # print(out.shape)

        print(out.sum()/batch_size)
        
        out2 = out.mean()
        print(out2)

        break 


if __name__ == "__main__":
    test_KL_divergence()
