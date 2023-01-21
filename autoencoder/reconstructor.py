import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from encoder import VariationalEncoder
from decoder import Decoder
from PIL import Image


# Hyper-parameters
BATCH_SIZE = 1
LATENT_SPACE = 95


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.model_file = os.path.join('autoencoder/model', 'var_autoencoder.pth')
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()
    
    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()


def main():

    data_dir = 'autoencoder/dataset/'

    test_transforms = transforms.Compose([transforms.ToTensor()])

    test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
    
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    model.load()
    count = 1
    with torch.no_grad(): # No need to track the gradients
        for x, _ in testloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Decode data
            x_hat = model(x)
            x_hat = x_hat.cpu()
            x_hat = x_hat.squeeze(0)
            transform = transforms.ToPILImage()

            # convert the tensor to PIL image using above transform
            img = transform(x_hat)

            image_filename = str(count) +'.png'
            img.save('autoencoder/reconstructed/'+image_filename)
            count +=1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nTerminating...')
