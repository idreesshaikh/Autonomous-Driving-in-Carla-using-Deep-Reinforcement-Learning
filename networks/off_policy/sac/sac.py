import os
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from parameters import SAC_LEARNING_RATE, SAC_CHECKPOINT_DIR, LATENT_DIM
from autoencoder.variational_autoencoder import VariationalEncoder

class CriticNetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(CriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(SAC_CHECKPOINT_DIR, model)
        
        self.conv_encoder = VariationalEncoder(LATENT_DIM)
        self.conv_encoder.load()
        self.conv_encoder.eval()
        for params in self.conv_encoder.parameters():
            params.requires_grad = False
        
        self.Linear = nn.Sequential(
            nn.Linear(200+4+self.n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.q = nn.Linear(64,1)

        self.optimizer = optim.Adam(self.parameters(), lr=SAC_LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, img, nav, action):
        img = img.view(-1,3,128,128)
        img = self.conv_encoder(img)

        img = img.view(-1, 200)
        nav = nav.view(-1, 4)
        observation = torch.cat((img, nav), -1)
        concatenated = torch.cat((observation, action.reshape(1, 2)), -1).float()
        action_value = self.Linear(concatenated)
        q = self.q(action_value)
        return q

    def save_checkpoint(self):
        print('\nCheckpoint saving')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('\nCheckpoint loading')
        self.load_state_dict(torch.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(ValueNetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(SAC_CHECKPOINT_DIR, model)
        
        self.conv_encoder = VariationalEncoder(LATENT_DIM)
        self.conv_encoder.load()
        self.conv_encoder.eval()
        for params in self.conv_encoder.parameters():
            params.requires_grad = False
        
        self.Linear = nn.Sequential(
            nn.Linear(200+4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.v = nn.Linear(64,1)

        self.optimizer = optim.Adam(self.parameters(), lr=SAC_LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, img, nav):
        img = img.view(-1,3,128,128)
        img = self.conv_encoder(img)
        img = img.view(-1, 200)
        nav = nav.view(-1, 4)
        action_value = self.Linear(torch.cat((img, nav), -1))
        v = self.v(action_value)
        return v

    def save_checkpoint(self):
        print('\nCheckpoint saving')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('\nCheckpoint loading')
        self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(SAC_CHECKPOINT_DIR, model)
        self.reparameterization_noise = 1e-6
        
        self.conv_encoder = VariationalEncoder(LATENT_DIM)
        self.conv_encoder.load()
        self.conv_encoder.eval()
        for params in self.conv_encoder.parameters():
            params.requires_grad = False
        
        self.Linear = nn.Sequential(
            nn.Linear(200+4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.mean = nn.Linear(64,self.n_actions)
        self.variance = nn.Linear(64,self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=SAC_LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, img, nav):
        img = img.view(-1,3,128,128)
        img = self.conv_encoder(img)
        img = img.view(-1, 200)
        nav = nav.view(-1, 4)
        prob = self.Linear(torch.cat((img, nav), -1))
        
        mean = self.mean(prob)
        variance = self.variance(prob)
        #variance = F.sigmoid(variance)
        variance = torch.clamp(variance, min=self.reparameterization_noise, max=1)

        return mean, variance

    def sample_normal(self,img_state, nav_state, reparameterize=True):
        mean, variance = self.forward(img_state, nav_state)
        probabilities = Normal(mean, variance)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        #print(actions.shape)
        #action = torch.tanh(actions).to(self.device)
        action1 = torch.tanh(actions[0][0]).to(self.device)
        action2 = torch.sigmoid(actions[0][1]).to(self.device).view(-1)
        #action = torch.cat((action1,action2), -1).to(self.device)
        action = torch.tensor([action1,action2]).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparameterization_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        print('\nCheckpoint saving')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('\nCheckpoint loading')
        self.load_state_dict(torch.load(self.checkpoint_file))

