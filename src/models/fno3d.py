import torch
import torch.nn as nn
import torch.nn.functional as F
from .spectral_conv3d import SpectralConv3d

class FNO3D(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channels=4, out_channels=5, n_layers=4):
        super(FNO3D, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.n_layers = n_layers

        self.p = nn.Linear(in_channels, width)
        
        self.convs = nn.ModuleList()
        self.ws = nn.ModuleList()
        
        for _ in range(n_layers):
            self.convs.append(SpectralConv3d(width, width, modes1, modes2, modes3))
            self.ws.append(nn.Conv3d(width, width, 1))

        self.q = nn.Linear(width, 128)
        self.q2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x is (B, C, X, Y, Z)
        # linear layer expects channels at the end
        x = x.permute(0, 2, 3, 4, 1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = F.gelu(x)
                
        x = x.permute(0, 2, 3, 4, 1)
        x = self.q(x)
        x = F.gelu(x)
        x = self.q2(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x
