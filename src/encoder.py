import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, out_nc=512):
        super().__init__()
        nc = out_nc // (2 ** 3)
        layers = [
            nn.Conv2d(3, nc, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(nc),
            nn.ReLU(),
        ]
        for i in range(1, 4):
            layers += [
                nn.Conv2d(nc, nc * 2, [3, 3], 2, 1, bias=False),
                nn.BatchNorm2d(nc * 2),
                nn.ReLU(),
            ]
            nc = 2 * nc
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
