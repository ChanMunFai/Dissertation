"""Architecture modified from https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import count_parameters

class Conv(nn.Module):
    """
    Convolutional layers to embed x_t (shape: Number of channels X Width X Height) to
    x_t_tilde (shape: h_dim)

    h_dim = 1024
    """
    def __init__(self, image_channels = 1):
        super(Conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, bias = False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Flatten()
            # shape: batch_size X 1024 (input size of 64 X 64)
        )

    def forward(self, input):
        return self.main(input)

class FastEncoder(nn.Module): 
    """ Embed x_t to x_t tilde 

    Modified from: https://github.com/charlio23/bouncing-ball/blob/main/models/modules.py
    """
    def __init__(self, input_channels, output_dim):
        super(FastEncoder, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=32,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.hidden_conv = nn.ModuleList([
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1)
        for _ in range(1)])

        self.out = nn.Linear(32*8*8, output_dim)
    
    def forward(self, x):
        B, NC, H, W = x.size()

        x = F.relu(self.in_conv(x))
        for hidden_layer in self.hidden_conv:
            x = F.relu(hidden_layer(x))
        x = x.flatten(-3, -1)
        
        xt_tilde = self.out(x)        
        return xt_tilde


class Conv_64(nn.Module):
    """
    Convolutional layers to embed x_t (shape: Number of channels X Width X Height) to
    x_t_tilde (shape: h_dim)

    h_dim = 64
    """
    def __init__(self, image_channels = 1):
        super(Conv_64, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, bias = False),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size = 5, stride = 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 5, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 5, stride = 2),
            nn.ReLU(),
            nn.Flatten()
            # shape: batch_size X 64 (input size of 64 X 64)
        )

    def forward(self, input):
        return self.main(input)


class UnFlatten(nn.Module):
    def forward(self, input):
        output = input.view(input.size(0), input.size(1), 1, 1)
        return output

class FastDecoder(nn.Module):
    """ Embed h_t-1 and z_t_tilde (combined dimension: 2 * h_dim)
    into x_t_hat (shape: Number of channels X Width X Height)

    Code modified from https://github.com/charlio23/bouncing-ball/blob/main/models/modules.py

    Arguments: 
        input_dim: dimension of latent variable h_t
        output_channels: typically 1 or 3 
        output_size: 28, 32 or 64
    """
    def __init__(self, input_dim, output_channels, output_size):
        super(FastDecoder, self).__init__()
        self.output_size = output_size 
        
        if self.output_size == 64: 
            self.latent_size = 16 
        elif self.output_size == 32: 
            self.latent_size = 8 
        elif self.output_size == 28: 
            self.latent_size = 7 
        else: 
            raise NotImplementedError

        self.output_channels = output_channels

        self.in_dec = nn.Linear(2*input_dim, 32*self.latent_size*self.latent_size)
        self.hidden_convs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=32,
                        out_channels=64,
                        kernel_size=3,
                        stride=2,
                        padding=1),
            nn.ConvTranspose2d(in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                        padding=1)])
        self.out_conv = nn.Conv2d(in_channels=32,
                        out_channels=output_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1)

    def forward(self, x):
        B, h_dim = x.size()
        h_dim = h_dim/2
        x = self.in_dec(x).reshape((B, -1, self.latent_size, self.latent_size))
        for hidden_conv in self.hidden_convs:
            x = F.relu(hidden_conv(x))
            x = F.pad(x, (0,1,0,1))

        x = torch.sigmoid(self.out_conv(x))
        x = torch.reshape(x, (B, self.output_channels, self.output_size, self.output_size))
        return x


class Deconv(nn.Module):
    """
    Deconvolutional (tranposed convolutional) layers to embed h_t-1 and z_t_tilde (combined dimension: 2 * h_dim)
    into x_t_hat (shape: Number of channels X Width X Height)
    """
    def __init__(self, image_channels = 1, h_dim = 1024):
        super(Deconv, self).__init__()
        self.main = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim * 2, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
            # shape: batch_size X number_channels X width X height
        )

    def forward(self, input):
        return self.main(input)

def test_conv():
    img = torch.zeros(10, 1, 64, 64) # batch size X number of channels X height X width
    print(img.shape)
    conv_encoder = Conv()
    output = conv_encoder(img)
    print(output.shape)

def test_conv_v2():
    img = torch.zeros(10, 1, 64, 64) # batch size X number of channels X height X width
    print(img.shape)
    conv_encoder = Conv_64()
    output = conv_encoder(img)
    print(output.shape)

def test_deconv():
    h_t = torch.zeros(10, 1024) # batch size X h_dim
    z_t_tilde = torch.zeros(10, 1024)
    input = torch.cat([h_t, z_t_tilde], 1)
    print(input.shape)
    decoder = Deconv()
    output = decoder(input)
    print(output.shape)

def test_fast_encoder(): 
    img = torch.rand(10, 1, 32, 32)
    encoder = FastEncoder(1, 20)
    xt_tilde = encoder(img)
    print(count_parameters(encoder))
    print(xt_tilde.shape)

def test_fast_decoder(): 
    h_t = torch.zeros(10, 16) # batch size X h_dim
    z_t_tilde = torch.zeros(10, 16)

    input = torch.cat([h_t, z_t_tilde], 1)
    decoder = FastDecoder(input_dim = 16, output_channels = 1, output_size = 32)
    output = decoder(input)
    print(count_parameters(decoder))
    print(output.shape)

if __name__ == "__main__":
    test_fast_encoder()
    test_fast_decoder()

