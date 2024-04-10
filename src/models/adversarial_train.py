# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from NN import SimpleBinaryClassificationNet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nn',help="Simple or Ensemble")
args = parser.parse_args()

# FGSM attack code
def fgsm_attack(input, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_data_ = input + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_data = torch.clamp(perturbed_data_, 0, 1)
    # Return the perturbed image
    return perturbed_data_


# Load Sample Training data
def load_training_data():
    X_train = torch.Tensor(np.load("../../data/external/X_train.npy"))
    X_train.requires_grad = True
    y_train = torch.Tensor(np.load("../../data/external/y_train.npy"))
    return X_train, y_train


def Adversarial_Simple_main():
    
    # Load Sample Training data
    X_train, y_train = load_training_data()
    
    # model
    torch.manual_seed(0)
    SimpleNet = SimpleBinaryClassificationNet()
    
    # optimizer and loss function
    opt = torch.optim.Adam(SimpleNet.parameters(), lr=0.001)
    loss_func = torch.nn.BCEWithLogitsLoss()

    # epochs
    epochs = 1000
    
    # epsilon for adversarial attack
    epsilon = 0.3
    
    # training
    SimpleNet.train()
    train_loss_list = []
    for epoch in tqdm(range(epochs)):
        # logits
        y_logits = SimpleNet(X_train).squeeze()
        
        loss = loss_func(y_logits, y_train)
        opt.zero_grad()
        loss.backward()
        # opt.step()
        
        perturbed_data = fgsm_attack(X_train,epsilon=epsilon,data_grad=X_train.grad)
    
        y_logits_perturbed = SimpleNet(perturbed_data).squeeze()
        loss_perturbed = loss_func(y_logits_perturbed, y_train)
        loss_perturbed.backward()
        #loss_ = loss_perturbed + loss
        #loss_.backward()
        
        opt.step()
        
        train_loss_list.append(loss_perturbed.item())
        
    # visualize training loss
    plt.plot([x for x in range(len(train_loss_list))], train_loss_list)
    plt.title("Adversarial training loss curve")
    plt.grid()
    plt.savefig("../../reports/figures/Adversarial Training loss curve.png")
    
    # save model
    torch.save(SimpleNet.state_dict(),"./weight/Adversarial_SimpleNet.pth")
    
def Adversarial_Ensemble_main():
    
    # Load Sample Training data
    X_train, y_train = load_training_data()
    
    # 5 ensembles by SimpleNN, optimizer define
    Simple_NN_Ensembles = []
    Simple_NN_Ensembles_opt = []
    for i in range(5):
        torch.manual_seed(i+1)
        net = SimpleBinaryClassificationNet()
        Simple_NN_Ensembles.append(net)
        Simple_NN_Ensembles_opt.append(torch.optim.Adam(net.parameters(),lr=0.001))
    
    # loss function
    loss_func = torch.nn.BCEWithLogitsLoss()
    
    # epochs
    epochs = 1000
    
    # epsilon for adversarial attack
    epsilon = 0.3

    # train 5 ensembles
    for j, Net in enumerate(Simple_NN_Ensembles):
        print("Training SimpleNN Ensemble",j+1)
        Net.train()
        for epoch in range(epochs):   
            # logits
            y_logits = Net(X_train).squeeze()
        
            Simple_NN_Ensembles_opt[j].zero_grad()
            loss = loss_func(y_logits, y_train)
            if epoch == 0:
                print("Initial loss:",loss.item())
            loss.backward()
            
            perturbed_data = fgsm_attack(X_train,epsilon=epsilon,data_grad=X_train.grad)
    
            y_logits_perturbed = Net(perturbed_data).squeeze()
            loss_perturbed = loss_func(y_logits_perturbed, y_train)
            loss_perturbed.backward()
            
            Simple_NN_Ensembles_opt[j].step()
        print("Final loss:",loss_perturbed.item())
        
    # save
    f = open('./weight/Adversarial_NN_Ensembles.pkl','wb')
    pickle.dump(Simple_NN_Ensembles,f)
    
    
if args.nn == "Simple":
    Adversarial_Simple_main()
elif args.nn == "Ensemble":
    Adversarial_Ensemble_main()
else:
    raise NotImplementedError