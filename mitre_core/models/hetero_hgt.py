from __future__ import annotations

import torch.nn as nn


class HeteroHGTBaseline(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        raise NotImplementedError("HGT baseline wiring is scaffolded but not yet implemented in MITRE-CORE V3.")


__all__ = ["HeteroHGTBaseline"]
