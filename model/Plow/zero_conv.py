import torch
import torch.nn as nn
import torch.nn.functional as F

class ZerosLinear(nn.Module):
    def __init__(self, in_channels, out_channels, logscale_factor=3):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.scale = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        output = self.linear(input)
        output = output * torch.exp(self.scale * 3)
        return output 

class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        output = F.pad(input, [1, 1, 1, 1], value=1)
        output = self.conv(output)
        output = output * torch.exp(self.scale * 3)

        return output