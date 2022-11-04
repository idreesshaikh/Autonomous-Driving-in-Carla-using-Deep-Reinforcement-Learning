import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
from parameters import PPO_CHECKPOINT_DIR, LATENT_DIM, PPO_LEARNING_RATE


class ActorNetwork(nn.Module):
    def __init__(self, action_dim, model):
        super(ActorNetwork, self).__init__()
        self.action_dim = action_dim
        self.checkpoint_file = os.path.join(PPO_CHECKPOINT_DIR, model)
        
        self.actor = nn.Sequential(
            nn.Linear(200+3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=PPO_LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, obs):

        """
			Runs a forward pass on the neural network.

			Parameters:
				img, nav - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        dist = self.actor(obs)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, model):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(PPO_CHECKPOINT_DIR, model)
        
        self.critic = nn.Sequential(
            nn.Linear(200+3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=PPO_LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, obs):

        """
			Runs a forward pass on the neural network.

			Parameters:
				img, nav - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        value = self.critic(obs)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

