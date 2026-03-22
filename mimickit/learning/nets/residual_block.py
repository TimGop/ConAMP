import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, width, activation_builder):
        """
        Emulates the 4-layer residual block from the scaling-crl codebase.
        activation_builder is a function that returns a new activation layer
        """
        super().__init__()
        layers = []
        for _ in range(4):
            layers.extend([
                nn.Linear(width, width),
                nn.LayerNorm(width),
                activation_builder()
            ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)