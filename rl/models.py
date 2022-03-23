import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, output_size):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            # (batch_size, input_size, 1)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        # (batch_size, out_channels, ???)
        self.final_layer = nn.LazyLinear(out_features=output_size)

    def forward(self, x):
        x = self.model(x)
        if len(x.shape) == 2:
            return self.final_layer(x.flatten(start_dim=0))
        else:
            return self.final_layer(x.flatten(start_dim=1))