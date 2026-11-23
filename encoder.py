import torch
from torch import nn


class SourceSink(nn.Module):
    def __init__(self, source_scale=None):
        super().__init__()
        if source_scale is None:
            source_scale = torch.zeros(241, 480)
        if not isinstance(source_scale, nn.Parameter):
            source_scale = nn.Parameter(
                torch.as_tensor(source_scale, dtype=torch.float32).detach().clone()
            )
        self.source_scale = source_scale
        self.encoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, pm2p5, blh, t2m, d2m, ssrd):
        x = torch.stack((pm2p5, blh, t2m, d2m, ssrd), dim=-1)
        x = x.to(self.encoder[0].weight)
        return self.source_scale * self.encoder(x).squeeze(-1)


class Diffusivity(nn.Module):
    def __init__(self, k_scale=None):
        super().__init__()
        if k_scale is None:
            k_scale = torch.zeros(241, 480)
        if not isinstance(k_scale, nn.Parameter):
            k_scale = nn.Parameter(
                torch.as_tensor(k_scale, dtype=torch.float32).detach().clone()
            )
        self.k_scale = k_scale
        self.encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, blh, t2m, d2m, ssrd):
        x = torch.stack((blh, t2m, d2m, ssrd), dim=-1)
        x = x.to(self.encoder[0].weight)
        return torch.nn.functional.softplus(self.k_scale * self.encoder(x).squeeze(-1))
