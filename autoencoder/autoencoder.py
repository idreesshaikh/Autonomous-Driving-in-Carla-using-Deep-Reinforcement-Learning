import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


# Hyper-parameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LATENT_SPACE = 400


torch.manual_seed(0)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):  
        super(VariationalEncoder, self).__init__()

        self.model_file = os.path.join('autoencoder/model', 'var_encoder_model_v2.pt')

        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),  # 63, 63
            nn.ReLU())

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32, 32
            nn.ReLU())#,

        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16, 16
            nn.ReLU())

        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 8, 8
            nn.ReLU())

        self.encoder_layer5 = nn.Sequential(
            nn.Conv2d(256, 512, 2, stride=2),  # 4, 4
            nn.ReLU())

        self.linear = nn.Sequential(
            nn.Linear(4*4*512, 1024),
            nn.ReLU())

        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = self.encoder_layer5(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu =  self.mu(x)
        sigma = torch.exp(self.sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

    def save(self):
        #print('Saving model...')
        torch.save(self.state_dict(), self.model_file)

    def load(self):
        #print('Loading model...')
        self.load_state_dict(torch.load(self.model_file))

class Decoder(nn.Module):
    
    def __init__(self, latent_dims):
        super().__init__()

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4 * 4 * 512),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 4, 4))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3,  stride=2,
                               padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, output_padding=1),
            nn.Sigmoid())
        
    def forward(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)
    
    def save(self):
        self.encoder.save()

def train(model, trainloader, optim):
    model.train()
    train_loss = 0.0
    for(x, _) in trainloader:
        # Move tensor to the proper device
        x = x.to(device)
        x_hat = model(x)
        loss = ((x - x_hat)**2).sum() + model.encoder.kl
        optim.zero_grad()
        loss.backward()
        optim.step()
        train_loss+=loss.item()
    return train_loss / len(trainloader.dataset)


def test(model, testloader):
    # Set evaluation mode for encoder and decoder
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in testloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = model.encoder(x)
            # Decode data
            x_hat = model(x)
            loss = ((x - x_hat)**2).sum() + model.encoder.kl
            val_loss += loss.item()

    return val_loss / len(testloader.dataset)


def main():

    data_dir = 'autoencoder/dataset/'

    writer = SummaryWriter(f"runs/"+"auto-encoder-v2")
    
    # Applying Transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.ImageFolder(data_dir+'train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir+'test', transform=test_transforms)
    
    m=len(train_data)
    train_data, val_data = random_split(train_data, [int(m-m*0.2), int(m*0.2)])
    

    # Data Loading
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

    #model = CNNVAutoencoder(latent_dims=latent_space).to(device)
    
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f'Selected device :) :) :) {device}')

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model,trainloader, optim)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch+1)
        val_loss = test(model,validloader)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch+1)
        print('\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, NUM_EPOCHS,train_loss,val_loss))
    
    model.save()
    #model.eval()
    #with torch.no_grad():

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nTerminating...')
