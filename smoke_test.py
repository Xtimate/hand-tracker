import torch
from model import HandNet, decode, simcc_loss

m = HandNet()
px, py = m(torch.randn(2, 3, 192, 192))
print(px.shape, py.shape)
print(decode(px, py).shape)
kp = torch.rand(2, 21, 2) * 191
print(simcc_loss(px, py, kp).item())
