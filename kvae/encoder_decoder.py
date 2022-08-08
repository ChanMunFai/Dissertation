## Encoder-decoder architecture 
# Ensures that suitable encoder decoder is in place for dataset 

import os
import math
import logging
import argparse
from pprint import pprint

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision 
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR

import matplotlib.pyplot as plt
from dataloader.moving_mnist import MovingMNISTDataLoader
from dataloader.healing_mnist import HealingMNISTDataLoader
from kvae.modules import KvaeEncoder, Decoder64, DecoderSimple, CNNFastEncoder  

import wandb

class EncoderDecoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EncoderDecoder, self).__init__()
        self.args = kwargs['args']
        self.device = self.args.device 
        
        if self.args.dataset == "MovingMNIST": 
            self.encoder = KvaeEncoder(input_channels=1, input_size = 64, a_dim = self.args.a_dim).to(self.device)
            self.decoder = DecoderSimple(input_dim = self.args.a_dim, output_channels = 1, output_size = 64).to(self.device)
            # self.decoder = Decoder64(a_dim = 2, enc_shape = [32, 7, 7], device = self.device).to(self.device) # change this to encoder shape
        
        elif self.args.dataset == "HealingMNIST_20": 
            self.encoder = CNNFastEncoder(1, self.args.a_dim).to(self.device)
            self.decoder = DecoderSimple(input_dim = self.args.a_dim, output_channels = 1, output_size = 32).to(self.device)

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
        
    def forward(self, x):
        (B,T,C,H,W) = x.size()

        # q(a_t|x_t)
        a_sample, a_mu, a_log_var = self._encode(x) 
        
        x_hat = self._decode(a_sample).reshape(B,T,C,H,W)
        x_mu = x_hat # assume they are the same for now

        # Calculate Bernoulli Loss 
        eps = 1e-5
        prob = torch.clamp(x_mu, eps, 1 - eps) # prob = x_mu 
        ll = x * torch.log(prob) + (1 - x) * torch.log(1-prob)
        ll = ll.mean(dim = 0).sum() 

        # Calculate MSE  
        calc_mse = nn.MSELoss(reduction = 'sum') # pixel-wise MSE 
        mse = calc_mse(x_mu, x)
        mse = mse/x.size(0)

        return -ll, mse 

    def reconstruct(self, input): 
        """ Reconstruct x_hat based on input x. 
        """
        (B, T, C, H, W) = input.size()

        with torch.no_grad(): 
            a_sample, _, _ = self._encode(input) 
            x_hat = self._decode(a_sample).reshape(B,T,C,H,W)

        return x_hat 

class EncDecTrainer:
    def __init__(self, state_dict_path = None, *args, **kwargs):
        self.args = kwargs['args']
        self.writer = SummaryWriter()
        print(self.args)

        self.model = EncoderDecoder(args = self.args).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.85)
    
        if state_dict_path: 
            state_dict = torch.load(state_dict_path, map_location = self.args.device)
            logging.info(f"Loaded State Dict from {state_dict_path}")
            
            self.model.load_state_dict(state_dict)

    def train(self, train_loader):
        n_iterations = 0
        logging.info(f"Starting Encoder-Decoder training for {self.args.epochs} epochs.")

        logging.info("Reconstruction Loss, MSE") # header for losses

        # Save a copy of data to use for evaluation 
        example_data, example_target = next(iter(train_loader))

        example_data = example_data[0].clone().to(self.args.device)
        example_data = (example_data - example_data.min()) / (example_data.max() - example_data.min())
        example_data = torch.where(example_data > 0.5, 1.0, 0.0).unsqueeze(0)

        if wandb_on: 
            reconstructions, ground_truth = self._plot_reconstructions(example_data)
            wandb.log({"Ground Truth": [ground_truth]})
            wandb.log({"Predictions": [reconstructions]})

        for epoch in range(self.args.epochs):

            print("Epoch:", epoch)
            running_recon = 0 # keep track of loss per epoch
            running_mse = 0 

            for data, _ in tqdm(train_loader):
                
                data = data.to(self.args.device)
                data = (data - data.min()) / (data.max() - data.min())
                data = torch.where(data > 0.5, 1.0, 0.0)

                #forward + backward + optimize
                self.optimizer.zero_grad()
                loss, mse = self.model(data)

                loss.backward()
                
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                metrics = {"train/reconstruction_loss": loss, 
                            "train/mse": mse,
                }
                if wandb_on: 
                    wandb.log(metrics)

                n_iterations += 1
                running_recon +=  loss.item() 
                running_mse += mse.item()

            training_recon = running_recon/len(train_loader)
            training_mse = running_mse/len(train_loader)
            current_lr = self.scheduler.get_last_lr()[0]

            print(f"Epoch: {epoch}\
                    \n Reconstruction Loss: {training_recon}\
                    \n MSE: {training_mse}")

            logging.info(f"{training_recon:.8f}, {training_mse:.8f}, {current_lr}")
           
            if epoch % self.args.save_every == 0:
                self._save_model(epoch)

            if epoch % 5 == 0: 
                if wandb_on: 
                    reconstructions, ground_truth = self._plot_reconstructions(example_data)
                    wandb.log({"Ground Truth": [ground_truth]})
                    wandb.log({"Predictions": [reconstructions]})

            if epoch % self.args.scheduler_step == 0 and epoch != 0: 
                self.scheduler.step() 

        logging.info("Finished training.")

        final_checkpoint = self._save_model(epoch)
        logging.info(f'Saved model. Final Checkpoint {final_checkpoint}')

    def _plot_reconstructions(self,target): 
        reconstructed = self.model.reconstruct(target)
        reconstructed = reconstructed.squeeze(0)
        target = target.squeeze(0)
        reconstructed_frames = torchvision.utils.make_grid(reconstructed,reconstructed.size(0))
        ground_truth_frames = torchvision.utils.make_grid(target,target.size(0))
        reconstructed_wandb = wandb.Image(reconstructed_frames)
        ground_truth_wandb = wandb.Image(ground_truth_frames)

        return reconstructed_wandb, ground_truth_wandb

    def _save_model(self, epoch):  
        checkpoint_path = f'saves/{self.args.dataset}/enc_dec/{self.args.subdirectory}/a_dim={self.args.a_dim}/'

        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        filename = f'enc_dec_state_dict_{epoch}.pth'
        checkpoint_name = checkpoint_path + filename

        torch.save(self.model.state_dict(), checkpoint_name)
        print('Saved model to ', checkpoint_name)
        
        return checkpoint_name

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "HealingMNIST_20", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall_20, BouncingBall_50, HealingMNIST_20]")
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--subdirectory', default="testing", type=str)
parser.add_argument('--model', default="Enc_Dec", type=str)

parser.add_argument('--a_dim', default=2, type=int)
parser.add_argument('--lstm_layers', default=1, type=int, 
                    help = "Number of LSTM layers. To be used only when alpha is 'rnn'.")

parser.add_argument('--clip', default=150, type=int)
parser.add_argument('--scale', default=0.3, type=float)

parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--learning_rate', default=0.007, type=float)
parser.add_argument('--scheduler_step', default=82, type=int, 
                    help = 'number of steps for scheduler. choose a number greater than epochs to have constant LR.')
parser.add_argument('--save_every', default=10, type=int) 
parser.add_argument('--wandb_on', default=None, type=str)

def main():
    seed = 128
    torch.manual_seed(seed)
    args = parser.parse_args()

    global wandb_on 
    wandb_on = args.wandb_on 
    if wandb_on: 
        if args.subdirectory == "testing":
            wandb.init(project="Testing")
        else: 
            wandb.init(project=f"Enc_Dec_{args.dataset}")

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.dataset == "MovingMNIST": 
        state_dict_path = None 
    elif args.dataset == "BouncingBall_20": 
        state_dict_path = None 
    elif args.dataset == "BouncingBall_50": 
        state_dict_path = None 
    elif args.dataset == "HealingMNIST_20": 
        state_dict_path = None 
       
    # set up logging
    log_fname = f'{args.model}_a_dim={args.a_dim}_epochs={args.epochs}.log'
    log_dir = f"logs/{args.dataset}/{args.model}/{args.subdirectory}/"
    log_path = log_dir + log_fname
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_path, filemode='w+', level=logging.INFO)
    logging.info(args)
    if wandb_on: 
        wandb.config.update(args)

    # Datasets
    if args.dataset == "MovingMNIST": 
        train_set = MovingMNISTDataLoader(root='dataset/mnist', train=True, download=True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=args.batch_size,
                    shuffle=True)

        val_set = MovingMNISTDataLoader(root='dataset/mnist', train=False, download=True)
        val_loader = torch.utils.data.DataLoader(
                    dataset=val_set,
                    batch_size=args.batch_size,
                    shuffle=True)

    elif args.dataset == "HealingMNIST_20": 
        train_set = HealingMNISTDataLoader('dataset/HealingMNIST/20/', train = True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)

        val_set = HealingMNISTDataLoader('dataset/HealingMNIST/20/', train = False)
        val_loader = torch.utils.data.DataLoader(
                    dataset=val_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)

    else: 
        raise NotImplementedError

    trainer = EncDecTrainer(state_dict_path= state_dict_path, args=args)
    trainer.train(train_loader)

if __name__ == "__main__":
    main()


def test(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default = "MovingMNIST", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall]")
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    args = parser.parse_args()

    enc_dec = EncoderDecoder(args = args)
    x_data = torch.rand((32, 10, 1, 64, 64))  # BS X T X NC X H X W

    loss, mse = enc_dec(x_data)
    

        