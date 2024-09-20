import torch
import torch.nn as nn
from .module import Block

class Plow(nn.Module):
    def __init__(
        self, in_channel=3, con_channel=2, n_flow=6, n_block=1, affine=True, conv_lu=True
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, con_channel, n_flow, affine=affine, conv_lu=conv_lu))
        self.blocks.append(Block(n_channel, con_channel, n_flow, split=False, affine=affine))

    def preprocessing(self, x, proj=None):
        if proj is None :
            x = x.permute(0, 2, 1).unsqueeze(-1)
        else :
            x = proj(x).permute(0, 2, 1).unsqueeze(-1)
        return x

    def forward(self, pose3d, pose2d):
        """
        pose3d      : [B, J, 3]
        condition   : [B, J, 2]
        """
        input = self.preprocessing(pose3d, None)        # [B, C, J, 1]
        condition = self.preprocessing(pose2d, None)    # [B, C, J, 1]
        out = input

        log_p_sum = 0
        logdet = 0
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out, condition)            # [B, C, J, 1]
            z_outs.append(z_new)                                        # [B, C, J, 1]
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, condition, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                pose3d_feat = self.preprocessing(z_list[-1], None)      # [B, C, J, 1]
                pose2d_feat = self.preprocessing(condition, None)       # [B, C, J, 1]
                input = block.reverse(pose3d_feat, pose2d_feat, pose3d_feat, reconstruct=reconstruct)

            else:
                pose3d_feat = self.preprocessing(z_list[-(i + 1)], None)      # [B, C, J, 1]
                pose2d_feat = self.preprocessing(condition, None)             # [B, C, J, 1]
                input = block.reverse(input, pose2d_feat, pose3d_feat, reconstruct=reconstruct)

        return input