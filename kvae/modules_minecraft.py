import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from utils import count_parameters

class MinecraftEncoder(nn.Module):
    """ 
    Arguments: 
        input_channels: 1 or 3
        input_size: 32 or 64 
        a_dim: int 

    Returns
        a_mu: BS X T X 2 
        a_log_var: BS X T X 2 
        encoder_shape: (tuple)
    """

    def __init__(self, input_channels = 1, input_size = 64, a_dim = 2):
        super(MinecraftEncoder, self).__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        
        self.a_dim = a_dim
        filters = 32
        self.encode = nn.Sequential(
                nn.Conv2d(input_channels, filters, 4, stride = 2), 
                nn.ReLU(), 
                nn.Conv2d(filters, filters * 2, 4, stride = 2), 
                nn.ReLU(), 
                nn.Conv2d(filters * 2, filters * 4, 4, stride = 2), 
                nn.ReLU(), 
                nn.Conv2d(filters * 4, filters * 8, 4, stride = 2), 
                nn.ReLU(),
                nn.Conv2d(filters * 8, self.a_dim, 2), 
                nn.ReLU(),
            )

        if self.input_channels == 3: 
            if self.input_size == 64: 
                self.log_var_out = nn.Linear(self.a_dim, self.a_dim) 
        else: 
            raise NotImplementedError 
        
    def forward(self, x):
        B, T, NC, H, W = x.size()
        x = torch.reshape(x, (B * T, NC, H, W))

        x = self.encode(x)
        x = torch.flatten(x, start_dim = 1)
        
        a_mu = x
        a_log_var = 0.1 * self.log_var_out(x)

        a_mu = torch.reshape(a_mu, (B, T, self.a_dim))
        a_log_var = torch.reshape(a_log_var, (B, T, self.a_dim))

        return a_mu, a_log_var, None 


class MinecraftDecoder(nn.Module):
    """ 
    """
    def __init__(self, a_dim, out_channels):
        super(MinecraftDecoder, self).__init__()
        self.a_dim = a_dim
        self.out_channels = out_channels

        filters = 32

        self.dense = nn.Linear(self.a_dim, 1024)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels = 1024, 
                out_channels = filters * 4, 
                kernel_size = 5), 
            nn.ReLU(),   
            nn.ConvTranspose2d(
                in_channels = filters * 4, 
                out_channels = filters * 2 , 
                kernel_size = 5, 
                stride = 2), 
            nn.ReLU(), 
            nn.ConvTranspose2d(
                in_channels = filters * 2, 
                out_channels = filters, 
                kernel_size = 6, 
                stride = 2), 
            nn.ReLU(), 
            nn.ConvTranspose2d(
                in_channels = filters, 
                out_channels = self.out_channels, 
                kernel_size = 6, 
                stride = 2), 
            nn.Tanh()
        )

    def forward(self, a_seq):
        B, T, a_dim = a_seq.size()
        a_seq = torch.reshape(a_seq, (-1, a_dim))

        a_seq = self.dense(a_seq)
        a_seq = a_seq.view(-1, a_seq.size(-1), 1, 1)
        
        a_seq = self.main(a_seq)   
        x_mu = torch.reshape(a_seq, (B, T, self.out_channels, 64, 64))
        
        return x_mu 

if __name__ == "__main__":
    encoder = MinecraftEncoder(input_channels = 3, input_size = 64, a_dim = 200)
    print("Number of parameters in Encoder", count_parameters(encoder)) #105472

    data = torch.rand(4, 100, 3, 64, 64)
    encoded, *_ = encoder(data)
    print(encoded.shape)

    decoder = MinecraftDecoder(a_dim = 200, out_channels = 3)
    print("Number of parameters in Decoder", count_parameters(decoder))
    decoded = decoder(encoded)
    print(decoded.shape)

