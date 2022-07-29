# Code from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/train_vqvae.py

import os
import math
import logging
import argparse
from pprint import pprint
from tqdm import tqdm
import numpy as np 

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
from vqvae.model_vqvae import VQVAE

import wandb 

class VQVAETrainer:
    def __init__(self, state_dict_path = None, *args, **kwargs):
        self.args = kwargs['args']
        self.writer = SummaryWriter()
        print(self.args)

        # Change out encoder and decoder 
        self.model = VQVAE(in_channel = 1).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.85)
    
        if state_dict_path: 
            state_dict = torch.load(state_dict_path, map_location = self.args.device)
            logging.info(f"Loaded State Dict from {state_dict_path}")
            
            self.model.load_state_dict(state_dict)

        self.criterion = nn.MSELoss()

    def train(self, train_loader):
        logging.info(f"Starting VQVAE training for {self.args.epochs} epochs.")
        logging.info("Train Loss") # header for losses

        latent_loss_weight = 0.25 # change to args later 

        for epoch in range(self.args.epochs):            
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            running_recon = 0
            running_latent_loss = 0 

            for img, _ in tqdm(train_loader):
                img = img.to(self.args.device)
                img = (img - img.min()) / (img.max() - img.min())

                self.optimizer.zero_grad()

                out, latent_loss = self.model(img)
                recon_loss = self.criterion(out, img)
                latent_loss = latent_loss.mean()
                loss = recon_loss + latent_loss_weight * latent_loss
                loss.backward()
       
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                metrics = {"train/train_loss": loss, 
                            "train/reconstruction_loss": recon_loss, 
                            "train/latent_loss": latent_loss}
                if wandb_on: 
                    wandb.log(metrics)

                running_loss += loss.item()
                running_recon +=  recon_loss.item() 
                running_latent_loss += latent_loss.item()
               
            training_loss = running_loss/len(train_loader)
            training_recon = running_recon/len(train_loader)
            training_latent_loss = running_latent_loss/len(train_loader)

            print(f"Epoch: {epoch}\
                    \n Train Loss: {training_loss}\
                    \n Reconstruction Loss: {training_recon}\
                    \n Latent Loss: {training_latent_loss}")

            logging.info(f"{training_loss:.8f}, {training_recon:.8f}, {training_latent_loss:.8f}")
           
            if epoch % self.args.save_every == 0:
                self._save_model(epoch)

        logging.info("Finished training.")

        final_checkpoint = self._save_model(epoch)
        logging.info(f'Saved model. Final Checkpoint {final_checkpoint}')

    def _save_model(self, epoch):  
        checkpoint_path = f'saves/{self.args.dataset}/vqvae/{self.args.subdirectory}/'

        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        filename = f'vqvae_state_dict_scale_{epoch}.pth'
        checkpoint_name = checkpoint_path + filename

        torch.save(self.model.state_dict(), checkpoint_name)
        print('Saved model to ', checkpoint_name)
        
        return checkpoint_name


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "MNIST", type = str)
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--subdirectory', default="testing", type=str)
parser.add_argument('--model', default="VQVAE", type=str)

parser.add_argument('--clip', default=150, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--save_every', default=10, type=int) 
parser.add_argument('--wandb_on', default=None, type=str)
parser.add_argument("--size", type=int, default=256, help = "Size of Image")
    
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
            wandb.init(project=f"VQVAE_{args.dataset}")

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    state_dict_path = None 

    # set up logging
    log_fname = f'{args.model}_epochs={args.epochs}.log'
    log_dir = f"logs/{args.dataset}/{args.model}/{args.subdirectory}/"
    log_path = log_dir + log_fname
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_path, filemode='w+', level=logging.INFO)
    logging.info(args)
    if wandb_on: 
        wandb.config.update(args)

    # Dataset 
    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
        ]
    )

    if args.dataset == "MNIST": 
        train_set = torchvision.datasets.MNIST(
                    root = "dataset/MNIST", 
                    train = True, 
                    transform = transform, 
                    download = True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)

    
    # img, label = next(iter(train_loader))
    # print(img.size())
    # print(label)

    trainer = VQVAETrainer(state_dict_path= state_dict_path, args=args)
    trainer.train(train_loader)

if __name__ == "__main__": 
    main()



