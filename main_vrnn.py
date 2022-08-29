# Used to test wandb functions 

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
from vrnn.model_vrnn import VRNN
from dataloader.moving_mnist import MovingMNISTDataLoader
from dataloader.bouncing_ball import BouncingBallDataLoader
from dataloader.dancing_mnist import DancingMNISTDataLoader
from dataloader.healing_mnist import HealingMNISTDataLoader
import wandb

class VRNNTrainer:
    def __init__(self, state_dict_path = None, *args, **kwargs):
        self.args = kwargs['args']
        self.writer = SummaryWriter()

        self.model = VRNN(self.args.xdim, self.args.hdim, self.args.zdim, self.args.nlayers).to(self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.85)
    
        if state_dict_path: 
            state_dict = torch.load(state_dict_path, map_location = self.args.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded state dict from {state_dict_path}")
            logging.info(f"Loaded state dict from {state_dict_path}")

    def train(self, train_loader):
        logging.info(f"Starting VRNN training for {self.args.epochs} epochs.")
        logging.info("Train Loss, KLD, MSE") 

        # Save a copy of data to use for evaluation 
        example_data, example_target = next(iter(train_loader))

        example_data = example_data[0].clone().to(self.args.device)
        example_data = (example_data - example_data.min()) / (example_data.max() - example_data.min())
        example_data = torch.where(example_data > 0.5, 1.0, 0.0).unsqueeze(0)

        example_target = example_target[0].clone().to(self.args.device)
        example_target = (example_target - example_target.min()) / (example_target.max() - example_target.min())
        example_target = torch.where(example_target > 0.5, 1.0, 0.0).unsqueeze(0)

        if wandb_on: 
            predictions_coloured, predictions_overlapped = self._plot_predictions(example_data, example_target)
            wandb.log({"Predictions": [predictions_coloured]})
            wandb.log({"Predictions (Overlapped)": [predictions_overlapped]})

            reconstructed_wandb, ground_truth_wandb = self._plot_reconstructions(example_data)
            wandb.log({"Ground Truth": [ground_truth_wandb]})
            wandb.log({"Reconstructions": [reconstructed_wandb]})

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            running_kld = 0
            running_recon = 0

            for data, _ in tqdm(train_loader):
                data = data.to(self.args.device)
                data = (data - data.min()) / (data.max() - data.min())
                data = torch.where(data > 0.5, 1.0, 0.0)

                #forward + backward + optimize
                self.optimizer.zero_grad()
                kld_loss, recon_loss, _ = self.model(data)
                loss = self.args.beta * kld_loss + recon_loss
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()

                metrics = {"train/train_loss": loss, 
                            "train/reconstruction_loss": recon_loss, 
                            "train/kld": kld_loss
                }
                if wandb_on: 
                    wandb.log(metrics)
        
                # forward pass
                # print(f"Loss: {loss}")
                # print(f"KLD: {kld_loss}")
                # print(f"Reconstruction Loss: {recon_loss}") # non-weighted by beta
                # print(f"Learning rate: {self.scheduler.get_last_lr()}") 

                learning_rate = self.scheduler.get_last_lr()

                running_loss += loss.item()
                running_kld += kld_loss.item()
                running_recon +=  recon_loss.item() # non-weighted by beta

            training_loss = running_loss/len(train_loader)
            training_kld = running_kld/len(train_loader)
            training_recon = running_recon/len(train_loader)

            print(f"Epoch: {epoch}\
                    \n Train Loss: {training_loss}\
                    \n KLD Loss: {training_kld}\
                    \n Reconstruction Loss: {training_recon}")
            logging.info(f"{training_loss:.8f}, {training_kld:.8f}, {training_recon:.8f}")

            if epoch % self.args.save_every == 0 and epoch!=0:
                self._save_model(epoch)

            if epoch % self.args.scheduler_step == 0 and epoch != 0: 
                self.scheduler.step() 

            if epoch % 5 == 0: 
                if wandb_on: 
                    predictions_coloured, predictions_overlapped = self._plot_predictions(example_data, example_target)
                    wandb.log({"Predictions": [predictions_coloured]})
                    wandb.log({"Predictions (Overlapped)": [predictions_overlapped]})

                    reconstructed_wandb, ground_truth_wandb = self._plot_reconstructions(example_data)
                    wandb.log({"Ground Truth": [ground_truth_wandb]})
                    wandb.log({"Reconstructions": [reconstructed_wandb]})
                    
        logging.info("Finished training.")
        final_checkpoint = self._save_model(epoch)
        logging.info(f'Saved model. Final Checkpoint {final_checkpoint}')


    def _save_model(self, epoch): 
        model_name = self.args.model.lower()
        checkpoint_path = f'saves/{self.args.dataset}/{self.args.model}/{self.args.subdirectory}/beta={self.args.beta}/'

        os.makedirs(checkpoint_path, exist_ok = True)
        filename = f"vrnn_state_dict_{epoch}.pth"
        checkpoint_name = checkpoint_path + filename

        torch.save(self.model.state_dict(), checkpoint_name)
        print('Saved model to ', checkpoint_name)
        
        return checkpoint_name

    def _plot_predictions(self, input, target):
        predicted = self.model.predict(input, target)
        predicted = predicted.squeeze(0).to(target.device)
        target = target.squeeze(0)

        print(predicted.shape, target.shape)

        # Different colours prediction 
        empty_channel = torch.full_like(predicted, 0)
        stitched_video = torch.cat((predicted, empty_channel, target), 1)
        stitched_frames = torchvision.utils.make_grid(stitched_video, stitched_video.size(0))    
        stitched_wandb = wandb.Image(stitched_frames)

        # overlapped prediction 
        predicted = torch.where(predicted > 0.5, 1.0, 0.0).to(target.device) # binarise 
        overlap = torch.where(predicted == target, predicted, torch.tensor(0, dtype=predicted.dtype, device = target.device))
        overlap_frames = torchvision.utils.make_grid(overlap, overlap.size(0)) 
        overlapped_wandb = wandb.Image(overlap_frames)   
    
        return stitched_wandb, overlapped_wandb

    def _plot_reconstructions(self,target): 
        reconstructed = self.model.reconstruct(target)
        reconstructed = reconstructed.squeeze(0)
        target = target.squeeze(0)
        reconstructed_frames = torchvision.utils.make_grid(reconstructed,reconstructed.size(0))
        ground_truth_frames = torchvision.utils.make_grid(target,target.size(0))
        reconstructed_wandb = wandb.Image(reconstructed_frames)
        ground_truth_wandb = wandb.Image(ground_truth_frames)

        return reconstructed_wandb, ground_truth_wandb


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--dataset', default = "BouncingBall_50", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall_20, BouncingBall_50]")
parser.add_argument('--model', default="VRNN", type=str)
parser.add_argument('--subdirectory', default="v1", type=str)

parser.add_argument('--xdim', default=32, type=int)
parser.add_argument('--hdim', default=64, type=int)
parser.add_argument('--zdim', default=16, type=int)
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--clip', default=150, type=int)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--scheduler_step', default=20, type=int, 
                    help = 'number of steps for scheduler. choose a number greater than epochs to have constant LR.')

parser.add_argument('--save_every', default=10, type=int) 
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--step_size', default = 30, type = int)
parser.add_argument('--wandb_on', default = None)

def main():
    seed = 128
    torch.manual_seed(seed)

    args = parser.parse_args()

    global wandb_on 
    wandb_on = args.wandb_on 
    if wandb_on: 
        wandb.init(project=f"{args.model}_{args.dataset}")
            
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.dataset == "DancingMNIST_20_v2": 
        state_dict_path = "saves/DancingMNIST_20_v2/VRNN/v3/beta=0.0/vrnn_state_dict_99.pth" 
    elif args.dataset == "BouncingBall_50": 
        state_dict_path = "saves/BouncingBall_50/VRNN/v3/beta=0.0/vrnn_state_dict_99.pth"  
    elif args.dataset == "HealingMNIST_20":
        state_dict_path = "saves/HealingMNIST_20/VRNN/v1/beta=0.0/vrnn_state_dict_199.pth"
    else: 
        state_dict_path = None 

    # set up logging
    log_fname = f'{args.model}_beta={args.beta}_epochs={args.epochs}.log'
    log_dir = f"logs/{args.dataset}/{args.model}/{args.subdirectory}/beta={args.beta}/"
    log_path = log_dir + log_fname
    os.makedirs(log_dir, exist_ok = True)
    logging.basicConfig(filename=log_path, filemode='w+', level=logging.INFO)
    logging.info(args)
    if wandb_on: 
        wandb.config.update(args)

    logging.info(args)
    print(args)

    # Datasets
    if args.dataset == "MovingMNIST": 
        train_set = MovingMNISTDataLoader(root='dataset/mnist', train=True, download=True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=args.batch_size,
                    shuffle=True)
    elif args.dataset == "BouncingBall_50": 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/50/train')
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)
    elif args.dataset == "DancingMNIST_20_v2": 
        train_set = DancingMNISTDataLoader('dataset/DancingMNIST/20/v2/', train = True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)
    elif args.dataset == "HealingMNIST_20": 
        train_set = HealingMNISTDataLoader('dataset/HealingMNIST/20/', train = True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)
    else: 
        raise NotImplementedError

    # Load in model
    if args.model == "VRNN":
        trainer = VRNNTrainer(state_dict_path=state_dict_path, args=args)

    trainer.train(train_loader)

if __name__ == "__main__":
    main()



