import os
import math
import logging
import argparse
from pprint import pprint
from tqdm import tqdm
import copy

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision 
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sv2p.cdna import CDNA 
from sv2p.model_sv2p import PosteriorInferenceNet, LatentVariableSampler
from scheduler import LinearScheduler

from dataloader.moving_mnist import MovingMNISTDataLoader
from dataloader.bouncing_ball import BouncingBallDataLoader
from dataloader.healing_mnist import HealingMNISTDataLoader
from dataloader.dancing_mnist import DancingMNISTDataLoader

import wandb 

class SV2PTrainer:
    """ Trains SV2P model. 

    Stages: 
        0: train (determistic) CDNA architecture only 
        1: update CDNA architecture to include Z from prior variables 
        2: Use Z from posterior but do not include KL divergence
        3. Use Z from posterior and include KL divergence 
    """
    def __init__(self, 
                state_dict_path_det = None, 
                state_dict_path_stoc = None,
                state_dict_path_posterior = None, 
                beta_scheduler = None,  
                *args, **kwargs):
        """
        Args: 
            state_dict_path_det: path for state dictionary for a deterministic model 
            state_dict_path_stoc: path for state dictionary for a stochastic model 
            state_dict_posterior: path for state dictionary for a posterior model 
            beta_scheduler: scheduler object to schedule the warmup of beta 

        The models are differently defined from the original paper. Here, 
        Deterministic model: cond channels = 0 (saved from Stage 0) 
        Stochastic model: cond channels = 1 (saved and used in all other Stages)
        """

        self.args = kwargs['args']
        self.writer = SummaryWriter()
        print(self.args)

        self.beta_scheduler = beta_scheduler
        assert state_dict_path_det == None or state_dict_path_stoc == None

        self.det_model = CDNA(in_channels = 1, cond_channels = 0,
            n_masks = 10).to(self.args.device) # deterministic
        if state_dict_path_det: 
            state_dict = torch.load(state_dict_path_det, map_location = self.args.device)
            self.det_model.load_state_dict(state_dict)

        self.stoc_model = CDNA(in_channels = 1, cond_channels = 1,
            n_masks = 10).to(self.args.device) # stochastic
        if state_dict_path_stoc: 
            state_dict = torch.load(state_dict_path_stoc, map_location = self.args.device)
            self.stoc_model.load_state_dict(state_dict)
            logging.info(f"Loaded State Dict from {state_dict_path_stoc}")
        elif state_dict_path_det: 
            self.load_stochastic_model() # load deterministic layers into stochastic model 
            logging.info(f"Loaded State Dict from {state_dict_path_det}")
        
        # Posterior network
        if self.args.dataset == "BouncingBall_50": 
            self.q_net = PosteriorInferenceNet(tbatch = 50).to(self.args.device) 
        elif self.args.dataset == "MovingMNIST": 
            self.q_net = PosteriorInferenceNet(tbatch = 10).to(self.args.device) 
        elif self.args.dataset == "DancingMNIST_20_v2" or self.args.dataset == "HealingMNIST_20": 
            self.q_net = PosteriorInferenceNet(tbatch = 20).to(self.args.device)
        else: 
            raise NotImplementedError
        self.sampler = LatentVariableSampler()

        if self.args.stage == 2 or self.args.stage == 3: 
            if not state_dict_path_posterior: 
                print("WARNING!: State dict for posterior is not loaded")
                logging.info("WARNING!: State dict for posterior is not loaded")

        if self.args.stage == 0: 
            self.optimizer = torch.optim.Adam(self.det_model.parameters(),
                                            lr=self.args.learning_rate)
        elif self.args.stage == 1: 
            self.optimizer = torch.optim.Adam(self.stoc_model.parameters(),
                                            lr=self.args.learning_rate)
        elif self.args.stage == 2 or self.args.stage == 3: # Stage 2 trains posterior to give good latents but is unregulated 
            self.optimizer = torch.optim.Adam(list(self.stoc_model.parameters()) + list(self.q_net.parameters()),
                                            lr=self.args.learning_rate)

        self.criterion = nn.MSELoss(reduction = 'sum').to(self.args.device) # image-wise MSE

    def _split_data(self, data):
        """ Splits sequence of video frames into inputs and targets

        Both have shapes (Batch_Size X Seq_Len–1 X
                            Num_Channels X Height X Width)

        data: Batch Size X Seq Length X Channels X Height X Width

        Inputs: x_0 to x_T-1
        Targets: x_1 to x_T
        """
        inputs = data[:, :-1, :, :, :]
        targets = data[:, 1:, :, :, :]

        assert inputs.shape == targets.shape
        return inputs, targets

    def train(self, train_loader):
        """ Trains SV2P model. 
        """

        logging.info(f"Starting SV2P training on Stage {self.args.stage} for {self.args.epochs} epochs.")
        if self.args.stage == 0: 
            logging.info("Train Loss") # header for losses
        else: 
            logging.info("Train Loss, KLD, MSE") # only Stage 3 uses KLD but we track KLD for the rest

        steps = 0

        # Save a copy of data to use for evaluation 
        example_data, example_unseen = next(iter(train_loader))

        example_data = example_data[0].clone().to(self.args.device)
        example_data = (example_data - example_data.min()) / (example_data.max() - example_data.min())
        example_data = torch.where(example_data > 0.5, 1.0, 0.0).unsqueeze(0)
        example_data = example_data.float()

        example_unseen = example_unseen[0].clone().to(self.args.device)
        example_unseen = (example_unseen - example_unseen.min()) / (example_unseen.max() - example_unseen.min())
        example_unseen = torch.where(example_unseen > 0.5, 1.0, 0.0).unsqueeze(0)
        example_unseen = example_unseen.float()

        # self.predict(example_data, example_unseen)
        if wandb_on: 
            predictions = self.plot_predictions(example_data, example_unseen) 
            wandb.log({"Predictions": [predictions]})

        for epoch in range(self.args.epochs):
            print("Epoch:", epoch)
            running_loss = 0 # keep track of loss per epoch
            running_kld = 0
            running_recon = 0

            for data, unseen in tqdm(train_loader):
                data = data.to(self.args.device).float()
                data = (data - data.min()) / (data.max() - data.min())
                data = torch.where(data > 0.5, 1.0, 0.0)
                
                self.optimizer.zero_grad(set_to_none=True)

                # Separate data into inputs and targets
                # inputs, targets are both of size: Batch Size X Seq Length - 1 X Channels X Height X Width
                inputs, targets = self._split_data(data)
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

                hidden = None
                recon_loss = 0.0
                total_loss = 0.0

                # Sample latent variable z from posterior - same z for all time steps
                mu, sigma = self.q_net(data) 
                z = self.sampler.sample(mu, sigma).to(self.args.device) # to be updated with time-variant z 
                
                prior_mean = torch.full_like(mu, 0).to(self.args.device)
                prior_std = torch.full_like(sigma, 1).to(self.args.device) # check if correct 

                if self.args.stage == 1: # use z from prior 
                    mu = prior_mean
                    sigma = prior_std
                
                p = torch.distributions.Normal(mu,sigma)
                q = torch.distributions.Normal(prior_mean,prior_std)

                kld_loss = torch.distributions.kl_divergence(p, q).sum()/self.args.batch_size
                # print("KLD Divergence is", kld_loss)

                # recurrent forward pass
                for t in range(inputs.size(1)):
                    x_t = inputs[:, t, :, :, :]
                    targets_t = targets[:, t, :, :, :] # x_t+1

                    if self.args.stage == 0: 
                        predictions_t, hidden, _, _ = self.det_model(
                                                x_t, hidden_states=hidden)

                    else: 
                        predictions_t, hidden, _, _ = self.stoc_model(
                                                    inputs = x_t,
                                                    conditions = z,
                                                    hidden_states=hidden)

                    loss_t = self.criterion(predictions_t, targets_t) # compare x_t+1 hat with x_t+1
                    recon_loss += loss_t/inputs.size(0) # image-wise MSE summed over all time steps

                # print("recon_loss", recon_loss)
                total_loss += recon_loss

                if self.args.stage == 3: 
                    beta_value = self.beta_scheduler.step()
                    total_loss += beta_value * kld_loss

                # print("Total loss after KLD", total_loss)
                    
                self.optimizer.zero_grad()
                total_loss.backward() 
                self.optimizer.step()

                if self.args.stage == 0: 
                    nn.utils.clip_grad_norm_(self.det_model.parameters(), self.args.clip)
                else: 
                    nn.utils.clip_grad_norm_(self.stoc_model.parameters(), self.args.clip)
                    
                running_loss += total_loss.item()

                metrics = {"train/train_loss": total_loss, 
                           "train/reconstruction_loss": recon_loss, 
                           "train/kld": kld_loss}

                if wandb_on: 
                    wandb.log(metrics)

            training_loss = running_loss/len(train_loader)
            training_kld = running_kld/len(train_loader)
            training_recon = running_recon/len(train_loader)

            if self.args.stage == 0: 
                print(f"Epoch: {epoch} \n Train Loss: {training_loss}")
                logging.info(f"{training_loss:.8f}")
            else:
                print(f"Epoch: {epoch}\
                        \n Train Loss: {training_loss}\
                        \n KLD Loss: {training_kld}\
                        \n Reconstruction Loss: {training_recon}")
                
                if self.args.stage != 3:     
                    logging.info(f"{training_loss:.8f}, {training_kld:.8f}, {training_recon:.8f}")
                else: # only stage 3 needs beta values
                    logging.info(f"{training_loss:.8f}, {training_kld:.8f}, {training_recon:.8f}, {beta_value:.8f}")

            if epoch % self.args.save_every == 0:
                self._save_model(epoch)

                if wandb_on: 
                    predictions = self.plot_predictions(example_data, example_unseen) 
                    wandb.log({"Predictions": [predictions]})


        logging.info('Finished training')
        
        self._save_model(epoch)
        logging.info('Saved model. Final Checkpoint.')

    def _save_model(self, epoch):
        if self.args.stage != 3:  
            checkpoint_path = f'saves/{self.args.dataset}/sv2p/stage{self.args.stage}/{self.args.subdirectory}/'
        else: 
            checkpoint_path = f'saves/{self.args.dataset}/sv2p/stage{self.args.stage}/{self.args.subdirectory}/final_beta={self.args.beta_end}/'

        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        if self.args.stage == 0 or self.args.stage == 1: 
            cdna_filename = f'sv2p_cdna_state_dict_{epoch}.pth'
            checkpoint_name_cdna = checkpoint_path + cdna_filename

            if self.args.stage == 0: 
                torch.save(self.det_model.state_dict(), checkpoint_name_cdna)
            elif self.args.stage == 1:
                torch.save(self.stoc_model.state_dict(), checkpoint_name_cdna)

            print('Saved model to '+checkpoint_name_cdna)
        else: 
            cdna_filename = f'sv2p_cdna_state_dict_{epoch}.pth'
            posterior_filename = f'sv2p_posterior_state_dict_{epoch}.pth'
            checkpoint_name_cdna = checkpoint_path + cdna_filename
            checkpoint_name_posterior = checkpoint_path + posterior_filename

            torch.save(self.stoc_model.state_dict(), checkpoint_name_cdna)
            torch.save(self.q_net.state_dict(), checkpoint_name_posterior)

            print('Saved CDNA model to '+checkpoint_name_cdna)
            print('Saved Posterior model to '+checkpoint_name_posterior)

    def copy_state_dict(self, model1, model2):
        """ Copies state dictionary from model 1 to model 2. 

        Used to copy state dict from determnistic model to stochastic model, 
        which has an additional channel for Z. 
        """

        params1 = model1.named_parameters()
        params2 = model2.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 != "u_lstm.conv4.weight" and name1 != "u_lstm.conv4.bias":
                dict_params2[name1].data.copy_(param1.data)

        def test_copying():
            f = open("param_model1.txt", "w")
            f.write("#### Model 1 Parameters ####")

            for name, param in model1.named_parameters():
                f.write("\n")
                f.write(str(name))
                f.write(str(param))
                f.write("\n")

            f.close()

            f = open("param_model2.txt", "w")
            f.write("#### Model 2 Parameters ####")

            for name, param in model2.named_parameters():
                f.write("\n")
                f.write(str(name))
                f.write(str(param))
                f.write("\n")
                
            f.close()

        # test_copying()

    def load_stochastic_model(self): 
        self.copy_state_dict(self.det_model, self.stoc_model)
    
    def predict(self, inputs, unseen): 
        """ Predicts future frames given some input. 

        N.B. unseen is known as target in other scripts, but 
        is named differently here as targets is already used 
        to name the internal targets to train SV2P. 
        """ 
        z = self.sampler.sample_prior((inputs.size(0), 1, 8, 8)).to(self.args.device)
        hidden = None 
        predicted_frames = torch.zeros(1, unseen.size(1), 1, 64, 64, device = self.args.device)

        total_len = inputs.size(1) + unseen.size(1)
        
        with torch.no_grad(): 
            for t in range(total_len):
                if t < inputs.size(1): # seen data
                    x_t = inputs[:, t, :, :, :]

                    if self.args.stage == 0: 
                        predictions_t, hidden, _, _ = self.det_model(
                                                x_t, hidden_states=hidden)
                    else: 
                        predictions_t, hidden, _, _ = self.stoc_model(inputs = x_t, conditions = z,
                                                        hidden_states=hidden)

                else: 
                    x_t = predictions_t # use predicted x_t instead of actual x_t
                    if self.args.stage == 0: 
                        predictions_t, hidden, _, _ = self.det_model(
                                                x_t, hidden_states=hidden)
                    else: 
                        predictions_t, hidden, _, _ = self.stoc_model(inputs = x_t, conditions = z,
                                                    hidden_states=hidden)

                    predicted_frames[:, t-inputs.size(1)] = predictions_t

        return predicted_frames

    def plot_predictions(self, input, unseen): 
        predicted = self.predict(input, unseen)
        predicted = predicted.squeeze(0)
        unseen = unseen.squeeze(0)

        empty_channel = torch.full_like(predicted, 0)
        stitched_video = torch.cat((predicted, empty_channel, unseen), 1)
        stitched_frames = torchvision.utils.make_grid(stitched_video, stitched_video.size(0))    
        stitched_wandb = wandb.Image(stitched_frames)

        return stitched_wandb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = "BouncingBall_50", type = str, 
                    help = "choose between [MovingMNIST, BouncingBall_50]")
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--model', default="cdna", type=str)
parser.add_argument('--stage', default=3, type=int)
parser.add_argument('--subdirectory', default="testing", type=str)

parser.add_argument('--save_every', default=5, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--clip', default=10, type=int)

parser.add_argument('--beta_start', default=0, type=float) # should not change generally
parser.add_argument('--beta_end', default=0.001, type=float)

parser.add_argument('--wandb_on', default=None, type=str)

def main():
    seed = 128
    torch.manual_seed(seed)
    EPS = torch.finfo(torch.float).eps # numerical logs

    args = parser.parse_args()

    global wandb_on 
    wandb_on = args.wandb_on 
    if wandb_on: 
        if args.subdirectory == "testing":
            wandb.init(project="Testing")
        else:  
            wandb.init(project=f"{args.model}_{args.dataset}_stage={args.stage}")
            
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.dataset == "BouncingBall_50": 
        state_dict_path_det = None 
        state_dict_path_stoc = "saves/BouncingBall_50/sv2p/stage2/v1/sv2p_cdna_state_dict_19.pth"
        state_dict_posterior = "saves/BouncingBall_50/sv2p/stage2/v1/sv2p_posterior_state_dict_19.pth"
    elif args.dataset == "HealingMNIST_20": 
        state_dict_path_det =  None 
        state_dict_path_stoc = "saves/HealingMNIST_20/sv2p/stage1/v1/sv2p_cdna_state_dict_29.pth"
        state_dict_posterior = None  
    elif args.dataset == "DancingMNIST_20_v2": 
        state_dict_path_det =  None 
        state_dict_path_stoc = "saves/DancingMNIST_20_v2/sv2p/stage1/v1/sv2p_cdna_state_dict_29.pth"
        state_dict_posterior = None  
    else: 
        raise NotImplementedError

    # Set up logging
    log_fname = f'{args.model}_stage={args.stage}_{args.epochs}.log'
    if args.stage == 3: 
        log_dir = f"logs/{args.dataset}/{args.model}/stage{args.stage}/{args.subdirectory}/finalB={args.beta_end}/"
    else: 
        log_dir = f"logs/{args.dataset}/{args.model}/stage{args.stage}/{args.subdirectory}/"

    log_path = log_dir + log_fname
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_path, filemode='w+', level=logging.INFO)
    logging.info(args)

    if wandb_on: 
        wandb.config.update(args)

    # Datasets
    if args.dataset == "MovingMNIST": 
        train_set = MovingMNISTDataLoader(root='dataset/mnist', train=True, download=False)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=args.batch_size,
                    shuffle=True)

        val_set = MovingMNISTDataLoader(root='dataset/mnist', train=False, download=False)
        val_loader = torch.utils.data.DataLoader(
                    dataset=val_set,
                    batch_size=args.batch_size,
                    shuffle=True)

    elif args.dataset == "BouncingBall_50": # use the 64 X 64 version 
        train_set = BouncingBallDataLoader('dataset/bouncing_ball/bigger_64/50/train')
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)

        val_set = BouncingBallDataLoader('dataset/bouncing_ball/bigger_64/50/val')
        val_loader = torch.utils.data.DataLoader(
                    dataset=val_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)

    elif args.dataset == "HealingMNIST_20": # use the 64 X 64 version 
        train_set = HealingMNISTDataLoader('dataset/HealingMNIST/bigger_64/20/', train = True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)

        val_set = HealingMNISTDataLoader('dataset/HealingMNIST/bigger_64/20/', train = False)
        val_loader = torch.utils.data.DataLoader(
                    dataset=val_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)

    elif args.dataset == "DancingMNIST_20_v2": # use the 64 X 64 version 
        train_set = DancingMNISTDataLoader('dataset/DancingMNIST/bigger_64/20/', train = True)
        train_loader = torch.utils.data.DataLoader(
                    dataset=train_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)

        val_set = DancingMNISTDataLoader('dataset/DancingMNIST/bigger_64/20/', train = False)
        val_loader = torch.utils.data.DataLoader(
                    dataset=val_set, 
                    batch_size=args.batch_size, 
                    shuffle=True)
                
    else: 
        raise NotImplementedError
        

    training_steps = len(train_loader) * args.epochs
    beta_scheduler = LinearScheduler(training_steps, args.beta_start, args.beta_end)

    trainer = SV2PTrainer(state_dict_path_det, state_dict_path_stoc, state_dict_posterior, beta_scheduler, args=args)  
    trainer.train(train_loader)
        
    logging.info(f"Completed {args.stage} training")

if __name__ == "__main__":
    main()



