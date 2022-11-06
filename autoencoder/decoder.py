import torch
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 9 * 4 * 256),
            nn.LeakyReLU()
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256,4,9))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4,  stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2),
            nn.Sigmoid())
        
    def forward(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x