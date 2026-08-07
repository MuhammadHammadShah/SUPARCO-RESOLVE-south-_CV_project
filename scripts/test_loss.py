import torch

from soilreg.losses import build_loss

criterion = build_loss("mse")

prediction = torch.randn(16, 6)

target = torch.randn(16, 6)

loss = criterion(prediction, target)

print(loss)