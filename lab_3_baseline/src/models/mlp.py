"""A multilayer perceptron on raw pixels.

TASK 2. Build `MLP`: a fully connected network that maps one flattened,
standardised photograph (`in_dim` grey levels) to one number in [0, 1].

* `hidden` is a list of layer widths from the config, for example [256, 64];
  build one Linear + nonlinearity + Dropout block per entry.
* `dropout` is the probability from the config. Where it goes, and whether
  it belongs after the last hidden layer, is your call.
* The output must lie in [0, 1]. `prior` is the mean target of the training
  rows; it is passed to you for a reason you have to find yourself.
* `forward` returns shape (batch,), not (batch, 1).

It trains in seconds on a CPU. That is the point of starting here: every
decision in it can be tested in the time it takes to read this docstring.
Count its parameters against the number of training photographs before you
decide how much dropout it needs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float = 0.3, prior: float = 0.5):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_dim

        for h_dim in hidden:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)
        self.activation = nn.Sigmoid()

        prior_clamped = min(max(float(prior), 1e-6), 1.0 - 1e-6)
        prior_logit = math.log(prior_clamped / (1.0 - prior_clamped))
        nn.init.constant_(self.head.bias, prior_logit)
        nn.init.zeros_(self.head.weight)
        
        #raise NotImplementedError("TASK 2: build the layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        out = self.backbone(x)
        out = self.activation(self.head(out))
        return out.squeeze(-1)
        raise NotImplementedError("TASK 2: one number in [0, 1] per row, shape (batch,)")
