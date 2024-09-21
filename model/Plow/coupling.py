import torch
import torch.nn as nn
import torch.nn.functional as F

from .actnorm import ActNorm

class Coupling(nn.Module):
    def __init__(self, in_channels, cond_channels, mid_channels):
        super(Coupling, self).__init__()
        split_channel = in_channels//2 # 1
        a_channel, b_channel = in_channels-split_channel, split_channel

        self.nn = NN(b_channel, cond_channels, mid_channels, 2 * a_channel)
        self.scale = nn.Parameter(torch.ones(a_channel, 1, 1))

    def forward(self, x, x_cond):
        """
        x       : [B, 3, 19, 1]
        x_cond  : [B, 2, 19, 1]
        """
        x_change, x_id = x.chunk(2, dim=1)

        st = self.nn(x_id, x_cond)  
        s, t = st.chunk(2, dim=1)  
        s = self.scale * torch.tanh(s)

        x_change = (x_change + t) * s.exp() 
        logdet = s.flatten(1).sum(-1)
        x = torch.cat((x_change, x_id), dim=1)

        return x, logdet
    
    def reverse(self, x, x_cond):
        x_change, x_id = x.chunk(2, dim=1) 

        st = self.nn(x_id, x_cond)  
        s, t = st.chunk(2, dim=1)   
        s = self.scale * torch.tanh(s)

        x_change = x_change * s.mul(-1).exp() - t
        x = torch.cat((x_change, x_id), dim=1)
        return x


class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, cond_channels, mid_channels, out_channels,
                 use_act_norm=False):
        super(NN, self).__init__()
        norm_fn = ActNorm if use_act_norm else nn.BatchNorm2d

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.in_condconv = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)
        nn.init.normal_(self.in_condconv.weight, 0., 0.05)

        self.mid_conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.mid_condconv1 = nn.Conv2d(cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.mid_conv1.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv1.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        self.mid_condconv2 = nn.Conv2d(cond_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv2.weight, 0., 0.05)
        nn.init.normal_(self.mid_condconv2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, x_cond):
        """
        x       : [B, 3, 19, 1]
        x_cond  : [B, 2, 19, 1]


        x (return) : [B, 3, 19, 1]
        """
        x = self.in_norm(x)
        x = self.in_conv(x) + self.in_condconv(x_cond)
        x = F.relu(x)

        x = self.mid_conv1(x) + self.mid_condconv1(x_cond)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_conv2(x) + self.mid_condconv2(x_cond)
        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        return x