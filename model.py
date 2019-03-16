import torch
import numpy as np
from torch import nn


class DQN_MODEL(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden=512, filters=[32,64,64]):
        """
        Declare and Initialize Deep Neural Net.
        params :
            state_size     - state space size.
            action_size    - action space size.
            seed           - random number generator seed value.
            hidden         - number of hidden units.
            filters        - convolution filters for ConvNet.
        """
        super(DQN_MODEL, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden
        self.filters = filters
        torch.manual_seed(seed)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.state_size, out_channels=self.filters[0], kernel_size=(8,8), stride=4),
            nn.ReLU(),
            # nn.BatchNorm1d(self.filters[0]),
            # OUTPUT : 20x20x32

            nn.Conv2d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=(4,4), stride=2),
            nn.ReLU(),
            # nn.BatchNorm1d(self.filters[1]),
            # OUTPUT : 9x9x64

            nn.Conv2d(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=(3,3), stride=1),
            nn.ReLU(),
            # nn.BatchNorm1d(self.filters[2]),
            # OUTPUT : 7x7x64
        )

        self.fc = nn.Sequential(
            nn.Linear(7*7*64, self.hidden_size),
            nn.ReLU(),

            nn.Linear(self.hidden_size, self.action_size)
        )


    def forward(self, state):
        """
        Takes current state as input and returns action selection probabilities.
        params :
            state - current enviroment state.
        """
        output = self.conv_layers(state)
        output = output.view(-1, 7*7*64)
        output = self.fc(output)
        return output
