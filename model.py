from scipy.special import kl_div
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class HandNet(nn.Module):
    def __init__(self, size=192, k=2, n_kp=21):
        super().__init__()
        self.size, self.k = size, k
        m = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
        self.backbone = m.features
        self.kp_conv = nn.Conv2d(576, n_kp, 1)
        hw = (size // 32) ** 2
        self.fc_x = nn.Linear(hw, size * k)
        self.fc_y = nn.Linear(hw, size * k)
        self.presence = nn.Linear(576, 1)

    def forward(self, x):
        feat = self.backbone(x)
        f = self.kp_conv(feat).flatten(2)
        return self.fc_x(f), self.fc_y(f), self.presence(feat.mean((2, 3)))

def _targets(coord, n_bins, k, sigma=6.0):
    bins = torch.arange(n_bins, device=coord.device, dtype=torch.float32)
    t = torch.exp(-0.5 * ((bins - coord.unsqueeze(-1) * k) / sigma) ** 2)
    return t / t.sum(-1, keepdim=True).clamp_min(1e-8)

def simcc_loss(px, py, kp, k=2, mask=None):
    tx = _targets(kp[..., 0], px.shape[-1], k)
    ty = _targets(kp[..., 1], py.shape[-1], k)
    l = (F.kl_div(px.log_softmax(-1), tx, reduction="none").sum((-1, -2))) + F.kl_div(py.log_softmax(-1), ty, reduction="none").sum((-1, -2))
    if mask is None:
        return l.mean()
    return (l * mask).sum() / mask.sum().clamp_min(1)


def decode(px, py, k=2):
    return torch.stack([px.argmax(-1), py.argmax(-1)], dim=-1).float() / k
