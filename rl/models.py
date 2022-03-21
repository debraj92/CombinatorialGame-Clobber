import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            # (batch_size, input_size)
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # (batch_size, hidden_size)
            nn.Linear(in_features=hidden_size, out_features=output_size),
            # (batch_size, output_size)
        )

    def forward(self, x):
        return self.model(x)
