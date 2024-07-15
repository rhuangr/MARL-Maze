import torch
import torch.nn as nn
from torch.optim import Adam

class simple_nn(nn.Module):
    def __init__(self, layer_sizes, activation = nn.ReLU()):
        super(simple_nn, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.activations.append(activation)
            
        self.optimizer = Adam(self.parameters(), lr = 0.0001)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        for i in range(len(self.layers) - 1):
            x = self.activations[i](self.layers[i](x))
        x = self.layers[-1](x)
        return x
