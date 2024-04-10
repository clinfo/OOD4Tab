# torch
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torch.distributions as dist
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import copy


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2a = nn.Linear(hidden_size, latent_dim)
        self.fc2b = nn.Linear(hidden_size, latent_dim)

    def encoder(self, x):
        # h1 = F.relu(self.fc1(x))
        h1 = F.selu(self.fc1(x))
        return self.fc2a(h1), self.fc2b(h1)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, input_size):
        mu, log_var = self.encoder(x.view(-1, input_size))
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.fc3 = nn.Linear(latent_dim, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def decoder(self, z):
        # h3 = F.relu(self.fc3(z))
        h3 = F.selu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        return self.decoder(x)


# for calc loss
def loss_function(reconstruction, x, mu, log_var, beta, input_size):
    reconstruction_loss = F.mse_loss(reconstruction, x.view(-1, input_size), reduction='sum')  # modified
    KL_loss = - 0.5 * torch.sum(1 + log_var - log_var.exp() - mu * mu)
    return reconstruction_loss + KL_loss * beta


def reconstruction_loss_function(reconstruction, x, input_size):
    reconstruction_loss = F.mse_loss(reconstruction, x.view(-1, input_size), reduction='sum')  # modified
    return reconstruction_loss


# add for calc cell reconstruction loss (RMSE)
def cell_reconstruction_loss_function(reconstruction, x, input_size):
    reconstruction_loss = F.mse_loss(reconstruction, x.view(-1, input_size), reduction='none')
    reconstruction_loss = torch.sqrt(reconstruction_loss)
    return reconstruction_loss
