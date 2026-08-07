import torch

from soilreg.models.backbones import build_resnet50

model = build_resnet50(
    num_outputs=6,
    pretrained=False,
)

x = torch.randn(8, 3, 224, 224)

y = model(x)

print(model)

print()

print("Output Shape:", y.shape)