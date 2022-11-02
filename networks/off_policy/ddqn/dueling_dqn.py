import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parameters import DQN_LEARNING_RATE, DQN_CHECKPOINT_DIR, LATENT_DIM
from autoencoder.variational_autoencoder import VariationalEncoder

class DuelingDQnetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(DuelingDQnetwork, self).__init__()
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(DQN_CHECKPOINT_DIR, model)
        
        self.conv_encoder = VariationalEncoder(LATENT_DIM)
        self.conv_encoder.load()
        
        
        self.conv_encoder.eval()
        for params in self.conv_encoder.parameters():
            params.requires_grad = False
        
        self.Linear = nn.Linear(200, 64)
        self.fc_nav = nn.Linear(4, 64)
        self.V = nn.Linear(64, 1)
        self.A = nn.Linear(64, self.n_actions)

        self.fc_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=DQN_LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, y):
        x = x.view(-1,3,128,128)
        z = self.conv_encoder(x)
        fcv = F.relu(self.Linear(z.view(-1, 200)))
        fcn = F.relu(self.fc_nav(y.view(-1, 4)))
        fc_net = self.fc_net(torch.cat((fcv, fcn), -1))
        V = self.V(fc_net)
        A = self.A(fc_net)
        return V, A

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

