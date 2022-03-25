import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(
        self,
        output_size,
        max_board_size,
        in_channels=1,
        out_channels=64,
    ):
        super(DQN, self).__init__()
        model = nn.ModuleList()

        if max_board_size >= 30:
            model.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=16,
                )
            )
            model.append(nn.ReLU(inplace=True))
            model.append(nn.Dropout(p=0.3))
            in_channels = out_channels
            out_channels = out_channels // 2

        if max_board_size >= 15:
            model.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=8,
                )
            )
            model.append(nn.ReLU(inplace=True))
            model.append(nn.Dropout(p=0.3))
            in_channels = out_channels
            out_channels = out_channels // 2

        model.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
            )
        )
        model.append(nn.ReLU(inplace=True))
        model.append(nn.Dropout(p=0.3))
        in_channels = out_channels
        out_channels = out_channels // 2

        model.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
            )
        )
        model.append(nn.ReLU(inplace=True))

        self.model = nn.Sequential(*model)
        self.final_layer = nn.LazyLinear(out_features=output_size)

    def forward(self, x):
        x = self.model(x)
        if len(x.shape) == 2:
            return self.final_layer(x.flatten(start_dim=0))
        else:
            return self.final_layer(x.flatten(start_dim=1))
