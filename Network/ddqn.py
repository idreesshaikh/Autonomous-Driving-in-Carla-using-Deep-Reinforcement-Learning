import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from parameters import LEARNING_RATE, CHECKPOINT_DIR


class DDQnetwork(nn.Module):
    def __init__(self, n_actions, model):
        super(DDQnetwork, self).__init__()

        self.checkpoint_file = os.path.join(CHECKPOINT_DIR, model)
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 48, (5, 5), stride=3),
            nn.ReLU(),
            nn.Conv2d(48, 64, (3, 3), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_visual = nn.Linear(2048, 276)
        self.fc_raw = nn.Linear(12, 24)

        self.fc_net = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 84),
            nn.ReLU(),
            nn.Linear(84, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, visual_obs, raw_data):
        visual_obs = visual_obs.view(-1, 3, 128, 128)
        conv = self.conv_net(visual_obs)
        fcv = F.relu(self.fc_visual(conv))
        fcr = F.relu(self.fc_raw(raw_data))
        return self.fc_net(torch.cat((fcv, fcr), -1))

    def save_checkpoint(self):
        print('\nCheckpoint saving')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('\nCheckpoint loading')
        self.load_state_dict(torch.load(self.checkpoint_file))
