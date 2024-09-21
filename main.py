import torch
from model.Plow.coupling import NN, Coupling

x = torch.rand((1, 3, 19, 1))
x_change, x_id = x.chunk(2, dim=1)  
x_cond = torch.rand((1, 2, 19, 1))
model_coupling = Coupling(in_channels=3, cond_channels=2, mid_channels=256)
model_nn = NN(in_channels=1, cond_channels=2, mid_channels=256, 
                out_channels=2)

x, ldj = model_coupling(x, x_cond)
print(x.shape)