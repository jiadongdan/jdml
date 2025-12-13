import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, in_ch=1, depth=17, features=64):
        super().__init__()
        layers = []
        # first layer
        layers += [nn.Conv2d(in_ch, features, 3, padding=1), nn.ReLU(inplace=True)]
        # middle layers
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1), nn.BatchNorm2d(features), nn.ReLU(inplace=True)]
        # last layer
        layers += [nn.Conv2d(features, in_ch, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.net(x)
        return x - residual  # denoised image
