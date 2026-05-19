from __future__ import annotations

import torch.nn as nn


class DGITrainer(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, *args, **kwargs):
        raise NotImplementedError("DGI training migration is scaffolded but not yet ported.")
