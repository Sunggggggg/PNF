import torch
import torch.nn as nn
import torch.nn.functional as F
from .actnorm import ActNorm

class Injection(nn.Module):
    def __init__(self, cond_channels, out_channels, mid_channels=256):
        super().__init__()
        self.nn = NN(cond_channels, mid_channels, 2*out_channels)

    def forward(self, x, x_cond):
        """
        x       : [B, 3, 19, 1]
        x_cond  : [B, 2, 19, 1]
        """
        log_s, t = self.nn(x_cond).chunk(2, 1) 
        s = F.sigmoid(log_s + 2)
        out = (x + t) * s

        logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)
        return out, logdet
    
    def reverse(self, x, x_cond):
        log_s, t = self.nn(x_cond).chunk(2, 1) 
        s = F.sigmoid(log_s + 2)
        input = x / s - t
        return input
    
class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, mid_channels, out_channels, use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.mid_conv1.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        """
        x       : [B, 3, 19, 1]
        """
        x = self.in_norm(x)
        x = self.in_conv(x)
        x = F.relu(x)

        x = self.mid_conv1(x)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_conv2(x)
        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        return x