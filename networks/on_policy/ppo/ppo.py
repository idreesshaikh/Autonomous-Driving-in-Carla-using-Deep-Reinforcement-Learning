import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
from parameters import PPO_CHECKPOINT_DIR, LATENT_DIM, PPO_LEARNING_RATE
from autoencoder.variational_autoencoder import VariationalEncoder


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(PPO_CHECKPOINT_DIR, model)
        
        self.conv_encoder = VariationalEncoder(LATENT_DIM)
        self.conv_encoder.load()
        self.conv_encoder.eval()
        for params in self.conv_encoder.parameters():
            params.requires_grad = False
        
        self.actor = nn.Sequential(
            nn.Linear(200+4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
            nn.Softmax(dim=-1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=PPO_LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, img, nav):
        img = img.view(-1,3,128,128)
        img = self.conv_encoder(img)
        img = img.view(-1, 200)
        nav = nav.view(-1, 4)
        state = torch.cat((img, nav), -1)
        
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        print('\nCheckpoint saving')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('\nCheckpoint loading')
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(CriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(PPO_CHECKPOINT_DIR, model)
        
        self.conv_encoder = VariationalEncoder(LATENT_DIM)
        self.conv_encoder.load()
        self.conv_encoder.eval()
        for params in self.conv_encoder.parameters():
            params.requires_grad = False
        
        self.critic = nn.Sequential(
            nn.Linear(200+4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=PPO_LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, img, nav):
        img = img.view(-1,3,128,128)
        img = self.conv_encoder(img)

        img = img.view(-1, 200)
        nav = nav.view(-1, 4)
        state = torch.cat((img, nav), -1)
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        print('\nCheckpoint saving')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('\nCheckpoint loading')
        self.load_state_dict(torch.load(self.checkpoint_file))

