import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parameters import LEARNING_RATE, CHECKPOINT_DIR


class DDQnetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(DDQnetwork, self).__init__()

        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(CHECKPOINT_DIR, model)
        self.conv_net = nn.Sequential(   
            nn.Conv2d(3, 32, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(4608,2048),
        )

        self.Linear1 = nn.Linear(2048, 200)
        self.Linear2 = nn.Linear(2048, 200)
        self.Linear3 = nn.Linear(200, 150)
        self.fc_nav = nn.Linear(4, 50)
        self.V = nn.Linear(150, 1)
        self.A = nn.Linear(150, self.n_actions)

        self.fc_net = nn.Sequential(
            nn.Linear(200, 150),
            nn.ReLU(),
        )

        self.Normal= torch.distributions.Normal(0, 1)
        self.Normal.loc = self.Normal.loc.cuda()
        self.Normal.scale = self.Normal.scale.cuda()
        self.Kullback_Leibler = 0

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, y):
        x = x.view(-1,3,128,128)
        x = self.conv_net(x)
        mu =  self.Linear1(x)
        sigma = torch.exp(self.Linear2(x))
        z = mu + sigma*self.Normal.sample(mu.shape)
        self.Kullback_Leibler = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()  
        fcv = F.relu(self.Linear3(z))
        fcn = F.relu(self.fc_nav(y))
        fc_net = self.fc_net(torch.cat((fcv, fcn), -1))
        V = self.V(fc_net)
        A = self.A(fc_net)
        return V, A

    def save_checkpoint(self):
        print('\nCheckpoint saving')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('\nCheckpoint loading')
        self.load_state_dict(torch.load(self.checkpoint_file))