from functional import seq
import numpy as np
import torch
from torch import nn
from torch.nn import Linear, Module, ModuleList
from torch.nn.init import uniform_
import torch.nn.functional as F


class Net(Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 layer_dims,
                 init_bound,
                 initializer,
                 last_activation):
        super(Net, self).__init__()

        self._initializer = initializer
        self._last_activation = last_activation

        _layer_dims = [in_dim] + layer_dims + [out_dim]
        self._layers = ModuleList(seq(_layer_dims[:-1])
                                    .zip(_layer_dims[1:])
                                    .map(lambda x: Linear(x[0], x[1]))
                                    .to_list())

        self._init_weights(init_bound)
    
    def _init_weights(self, bound):
        # Initialize all layers except the last one with fan-in initializer
        (seq(self._layers[:-1])
            .map(lambda x: x.weight)
            .for_each(self._initializer))
        # Init last layer with uniform initializer
        uniform_(self._layers[-1].weight, -bound, bound)

    def forward(self, inp):
        out = torch.flatten(inp)

        for layer in self._layers[:-1]:
            out = F.relu(layer(out))

        if self._last_activation:
            out = self._last_activation(self._layers[-1](out))
        else:
            out = self._layers[-1](out)

        return out


class ConvNet(Module):
    def __init__(self,
                 obs_space,
                 out_dim,
                 layer_dims,
                 init_bound,
                 initializer,
                 last_activation):
        super(ConvNet, self).__init__()

        self._initializer = initializer
        self._last_activation = last_activation

        n_input_channels = 1 #obs_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.unsqueeze(torch.unsqueeze(torch.as_tensor(obs_space.sample()).float(), 0), 0)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, out_dim), nn.ReLU())

        # self._init_weights(init_bound)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = torch.unsqueeze(observations, 1)
        return self.linear(self.cnn(observations))

    def _init_weights(self, bound):
        # Initialize all layers except the last one with fan-in initializer
        (seq(self._layers[:-1])
         .map(lambda x: x.weight)
         .for_each(self._initializer))
        # Init last layer with uniform initializer
        uniform_(self._layers[-1].weight, -bound, bound)
