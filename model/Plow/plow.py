import torch
import torch.nn as nn
from .module import Block

class Plow(nn.Module):
    def __init__(
        self, in_channel=3, hidden_channel=32, n_flow=6, n_block=1, affine=True, conv_lu=True, condition=True
    ):
        super().__init__()
        self.pos_embed = nn.Linear(in_channel, hidden_channel)
        if condition :
            self.condition_embed = nn.Linear(2, hidden_channel)

        self.blocks = nn.ModuleList()
        n_channel = hidden_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, pose3d, condition=None):
        """
        pose3d      : [B, J, 3]
        condition   : [B, J, 2]
        """
        pose_feat = self.pos_embed(pose3d).permute(0, 2, 1).unsqueeze(-1)   # [B, C, J, 1]
        condition = self.condition_embed(condition).permute(0, 2, 1).unsqueeze(-1)
        out = pose_feat

        log_p_sum = 0
        logdet = 0
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out, condition)     # [B, C, J, 1]
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, condition, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input