import torch
from model.Plow import Plow

pose3d = torch.rand((1, 19, 3))
pose2d = torch.rand((1, 19, 2))
model = Plow()
trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(trainable_parameters)

model(pose3d, pose2d)
y = model.reverse([pose3d, pose3d, pose3d], pose2d)
print(y.shape)