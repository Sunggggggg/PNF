import torch
import torch.nn as nn

class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))
            std = (flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3))

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
            self.initialized.fill_(1)

    def forward(self, input):
        _, _, h, w = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            
        x = input + self.loc    # Center
        x = self.scale * x      # Scale
        
        log_abs = torch.log(torch.abs(self.scale))
        logdet = h * w * torch.sum(log_abs)
        return x, logdet

    def reverse(self, output):
        x = output / self.scale
        x = x - self.loc
        return x