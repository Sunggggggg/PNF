import torch
from torch import nn
from math import log, pi
from .actnorm import ActNorm
from .invconv import InvConv2dLU
from .inject import Injection
from .coupling import Coupling
from .zero_conv import ZeroConv2d

logabs = lambda x: torch.log(torch.abs(x))

class Flow(nn.Module):
    def __init__(self, in_channels=3, cond_channels=2, mid_channels=256):
        super().__init__()
        self.norm = ActNorm(in_channels)
        self.conv = InvConv2dLU(in_channels)
        self.inject = Injection(cond_channels, in_channels, mid_channels)
        self.coup = Coupling(in_channels, cond_channels, mid_channels)

    def forward(self, input, x_cond=None):
        out, logdet = self.norm(input)  
        out, det1 = self.conv(out)
        out, det2 = self.inject(out, x_cond)
        out, det3 = self.coup(out, x_cond)

        logdet = logdet + det1 + det2 + det3
        return out, logdet

    def reverse(self, output, x_cond):
        input = self.coup.reverse(output, x_cond)
        input = self.inject.reverse(output, x_cond)
        input = self.conv.reverse(input)
        input = self.norm.reverse(input)

        return input

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

class Block(nn.Module):
    def __init__(self, in_channels, cond_channels, mid_channels, n_flow, split=False):
        super().__init__()
        self.split = split

        self.flows = nn.ModuleList([
            Flow(in_channels, cond_channels, mid_channels) 
            for i in range(n_flow)]
        )

        split_channel = int(in_channel/2 + 0.5)
        self.prior = ZeroConv2d(split_channel, (in_channel-split_channel)*2)

    def forward(self, input, x_cond):
        B, C, J, _ = input.shape
        out = input

        logdet = 0.
        for flow in self.flows:
            out, det = flow(out, x_cond)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)                    # [B, 2, 19, 1], [B, 1, 19, 1]
            mean, log_sd = self.prior(out).chunk(2, 1)      # [B, 1, J, 1], [B, 1, J, 1]
            log_p = gaussian_log_p(z_new, mean, log_sd)     # [B, 1, J, 1]
            log_p = log_p.view(B, -1).sum(1)                # [B]
        else:
            zero = torch.zeros_like(out)                    # [B, 3, 19, 1]
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(B, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, condition, eps=None, reconstruct=False):
        input = output

        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)

            else:
                input = eps

        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)

            else:
                zero = torch.zeros_like(input)
                # zero = F.pad(zero, [1, 1, 1, 1], value=1)
                mean, log_sd = self.prior(zero).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input, condition)

        B, C, N, _ = input.shape

        unsqueezed = input
        return unsqueezed