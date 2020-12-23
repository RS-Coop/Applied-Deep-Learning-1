import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchsparse.nn as spnn
from torchsparse.nn.functional import spact


class sparseGRUCell(nn.module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #Initialize sparse convolutions
        input_channels = self.input_size + self.hidden_size

        #Note we could set bias to be true also
        self.reset_gate = spnn.Conv3d(input_channels, hidden_size, kernel_size=3, stride=1)
        self.update_gate = spnn.Conv3d(input_channels, hidden_size, kernel_size=3, stride=1)
        self.out_gate = spnn.Conv3d(input_channels, hidden_size, kernel_size=3, stride=1)

        #Letting torchsparse initialize the weights however they do

    def forward(self, input, h_prev):

        #Get batch and spatial sizes

        #Generate empty prev_state if None is provided
        if h_prev is None:
            state_size = 
            h_prev = torch.zeros(, device=input.device, requires_grad=True)

        stacked_inputs = torch.cat([input, h_prev], dim=1)
        update = spact(self.update_gate(), F.sigmoid)
        reset = spact(self.rest_gate(), F.sigmoid)

        candidate_input = torch.cat([input, h_prev*reset], dim=1)
        candidate = spact(self.out_gate(candidate_input), F.tanh)

        h_new = h_prev*update  + candidate*(1-update)

        return h_new

class sparseGRU(nn.module):
    def __init__(self, input_size, hidden_sizes, num_layers):
        super().__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*num_layers
        else:
            assert len(hidden_sizes) == num_layers
            self.hidden_sizes = hidden_sizes

        self.num_layers = num_layers

        cells = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i])
            name = 'sparseGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, h=None):
        if not hidden:
            h = [None]*self.num_layers

        h_new = []

        for idx in range(self.num_layers):
            cell = self.cells[idx]
            cell_h = h[idx]

            #Pass through layer
            cell_h_new = cell(x, cell_h)
            h_new.append(cell_h_new)

            #Update input_ to the last updated hidden layer for next pass
            x = cell_h_new

        #Retain tensors in list to allow different hidden sizes
        return h_new

##################################################################

class sparseLSTMCell():
    pass

class sparseLSTM():
    pass