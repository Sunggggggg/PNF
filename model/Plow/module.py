import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la
from .zero_init_module import *

logabs = lambda x: torch.log(torch.abs(x))

class AffineInjection(nn.Module):
    def __init__(self, in_channel, out_channel, filter_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, out_channel*2),
        )

    def forward(self, input, condition):
        log_s, t = self.net(condition).chunk(2, 1)  # [B, C/2, J, 1]
        s = F.sigmoid(log_s + 2)
        out = (input + t) * s

        logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        return out, logdet
    
    def reverse(self, out, condition):
        log_s, t = self.net(condition).chunk(2, 1)  # [B, 3, J, 1]
        s = F.sigmoid(log_s + 2)
        input = out / s - t
        return input

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, in_channel, con_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        n_channel = in_channel - in_channel//2  # 3, 1
        self.net = nn.Sequential(
            nn.Conv2d(n_channel + con_channel, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel//2 * 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, pose3d, condition=None):
        """
        pose_feat : [B, C, J, 1]
        condition : [B, C, J, 1]
        """
        in_a, in_b = pose3d.chunk(2, 1)             # [B, 2, J, 1], [B, 1, J, 1]

        if self.affine:
            z = torch.cat([in_a, condition], dim=1) # [B, 4, J, 1]
            log_s, t = self.net(z).chunk(2, 1)      # [B, 2, J, 1]
            s = F.sigmoid(log_s + 2)
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(pose3d.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet
    
    def reverse(self, output, condition=None):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            z = torch.cat([out_a, condition], dim=1) # [B, C/2+C, J, 1]
            log_s, t = self.net(z).chunk(2, 1)       # [B, C/2, J, 1]
            s = F.sigmoid(log_s + 2)
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)

class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, pose_feat):
        _, _, num_joint, _ = pose_feat.shape

        if self.initialized.item() == 0:
            self.initialize(pose_feat)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = num_joint * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (pose_feat + self.loc), logdet

        else:
            return self.scale * (pose_feat + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc

class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )

class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p.copy())
        w_l = torch.from_numpy(w_l.copy())
        w_s = torch.from_numpy(w_s.copy())
        w_u = torch.from_numpy(w_u.copy())

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, pose_feat):
        _, _, num_joint, _ = pose_feat.shape

        weight = self.calc_weight()

        out = F.conv2d(pose_feat, weight)
        logdet = num_joint * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, pose_feat):
        """
        pose_feat : [B, C, J, 1]
        """
        in_a, in_b = pose_feat.chunk(2, 1)  # [B, C/2, J, 1], [B, C/2, J, 1]

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)   # [B]
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(pose_feat.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)

class Flow(nn.Module):
    def __init__(self, in_channel, con_channel, affine=True, conv_lu=True, condition=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        
        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)

        if condition :
            self.coupling = ConditionalAffineCoupling(in_channel, con_channel, affine=affine)
            self.injection = AffineInjection(con_channel, in_channel)
        else:
            self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, pose_feat, condition=None):
        """
        pose_feat   : [B, C, J, 1]
        out         : [B, C, J, 1]
        """
        out, logdet = self.actnorm(pose_feat)               # [B, C, J, 1]
        out, det1 = self.invconv(out)                       # [B, C, J, 1]
        out, det2 = self.injection(out, condition)
        out, det3 = self.coupling(out, condition)           # [B, C, J, 1]

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        if det3 is not None:
            logdet = logdet + det3

        return out, logdet

    def reverse(self, output, condition):
        input = self.coupling.reverse(output, condition)
        input = self.injection.reverse(output, condition)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, con_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(in_channel, con_channel, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel//2, in_channel)
        else:
            self.prior = ZeroConv2d(in_channel, in_channel*2)

    def forward(self, pose_feat, condition):
        """
        pose_feat   : [B, 3, J, 1]
        out         : [B, 3, J, 1]
        logdet      : [B]
        log_p       : [B]
        z_new       : [B, C/2, J, 1]
        """
        B, C, J, _ = pose_feat.shape
        out = pose_feat

        logdet = 0
        for flow in self.flows:
            out, det = flow(out, condition)        # [B, C, J, 1]
            logdet = logdet + det       # 

        if self.split:
            out, z_new = out.chunk(2, 1)                    # [B, C/2, J, 1]
            mean, log_sd = self.prior(out).chunk(2, 1)      # [B, C/2, J, 1], [B, C/2, J, 1]
            log_p = gaussian_log_p(z_new, mean, log_sd)     # [B, C/2, J, 1]
            log_p = log_p.view(B, -1).sum(1)                # [B]
        else:
            zero = torch.zeros_like(out)
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
    

if __name__ == '__main__':
    x = torch.rand((1, 256, 19, 1))
    model = Flow(in_channel=256)
    out, logdet = model(x)
    print(out.shape)