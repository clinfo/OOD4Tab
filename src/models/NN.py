# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBinaryClassificationNet(nn.Module):
    
    def __init__(self):

        super(SimpleBinaryClassificationNet, self).__init__()

        # nn.Sequential
        # First Layer: input features:2，output features:32
        self.layers1 = nn.Sequential(
            nn.Linear(in_features=2, out_features=32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        # Second Layer: input features:32，output features:16
        self.layers2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            # nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        # Final Layer: output features:1
        self.layers_out = nn.Linear(in_features=16, out_features=1)
    
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers_out(x)
        return x