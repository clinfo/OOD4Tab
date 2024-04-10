# -*- coding: utf-8 -*-
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from vae import Encoder
from vae import Decoder
from vae_loss import train
from vae_loss import test

#%%
def load_training_data():
    # Load Sample Training data
    X_train = torch.Tensor(np.load("../../data/external/X_train.npy"))
    X_train.requires_grad = True
    y_train = torch.Tensor(np.load("../../data/external/y_train.npy"))
    return X_train, y_train


def main():
    
    X_train, y_train = load_training_data()
    
    input_size = 2
    epochs = 200
    
    torch.manual_seed(0)
    
    netEncoder = Encoder(input_size = 2, hidden_size = 10, latent_dim = 2)
    netDecoder = Decoder(input_size = 2, hidden_size = 10, latent_dim = 2)
    
    optimizer1 = optim.Adam(netEncoder.parameters(), lr=0.001)
    optimizer2 = optim.Adam(netDecoder.parameters(), lr=0.001)
    
    train_loss = 0
    val_loss = 0
    train_loss_list = []
    val_loss_list = []
    num_train = X_train.shape[0]
    
    #num_val = val_index.shape[0]

    # 学習開始
    for epoch in tqdm(range(1, epochs+1)):
        train_loss = train(netEncoder, netDecoder, X_train, optimizer1, optimizer2, num_train,
                        1, input_size)
        #val_loss = test(netEncoder, netDecoder, val_dataloader, num_val, vae_param["beta"], input_size)
        train_loss_list.append(train_loss)
        #val_loss_list.append(val_loss)

    plt.figure()
    x_list = [x for x in range(epochs)]
    plt.plot(x_list,train_loss_list)
    plt.grid()
    
    torch.save(netEncoder.state_dict(), "./weight/vae_encoder.pth")
    torch.save(netDecoder.state_dict(), "./weight/vae_decoder.pth")

if __name__ == '__main__':
    main()
# %%
