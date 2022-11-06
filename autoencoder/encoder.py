import os
import torch
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):  
        super(VariationalEncoder, self).__init__()

        self.model_file = os.path.join('autoencoder/model', 'var_encoder_model.pth')

        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # 79, 39
            nn.LeakyReLU())

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 40, 20
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2),  # 19, 9
            nn.LeakyReLU())

        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2),  # 9, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.linear = nn.Sequential(
            nn.Linear(9*4*256, 1024),
            nn.LeakyReLU())

        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu =  self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        self.load_state_dict(torch.load(self.model_file))