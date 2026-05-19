from __future__ import annotations

import torch.nn as nn


class MaskedAutoencoderTrainer(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Masked autoencoder training migration is scaffolded but not yet ported.")
